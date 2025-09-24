import os.path
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import partial
from collections import OrderedDict

from datetime import datetime
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CalciumDataset_one_stream_pair_pca(Dataset):
    """Calcium imaging dataset with integrated PCA and data augmentation - One stream version"""

    def __init__(self, data_dict_list, seq_length=196, pca_dim=128, save_path=None,
                 training=True,
                 # Augmentation parameters
                 random_crop=True,
                 channel_shuffle_prob=0.2,
                 channel_shuffle_ratio=0.1,
                 channel_dropout_prob=0.2,
                 channel_dropout_ratio=0.1,
                 time_mask_prob=0.2,
                 time_mask_ratio=0.1):

        self.seq_length = seq_length
        self.pca_dim = pca_dim
        self.save_path = save_path
        self.training = training

        # Augmentation settings
        self.random_crop = random_crop
        self.channel_shuffle_prob = channel_shuffle_prob
        self.channel_shuffle_ratio = channel_shuffle_ratio
        self.channel_dropout_prob = channel_dropout_prob
        self.channel_dropout_ratio = channel_dropout_ratio
        self.time_mask_prob = time_mask_prob
        self.time_mask_ratio = time_mask_ratio

        # Initialize storage
        self.processed_trials = []
        self.win_labels = []
        self.trial_lengths = []  # Store original trial lengths for analysis
        self.pca_stats = []  # Store PCA statistics

        # Check for cached data
        if save_path is not None and os.path.exists(save_path):
            print(f"Loading preprocessed data from {save_path}...")
            with open(save_path, 'rb') as f:
                cache = pickle.load(f)
                self.processed_trials = cache['processed_trials']
                self.win_labels = cache['win_labels']
                self.trial_lengths = cache.get('trial_lengths', [])
                self.pca_stats = cache.get('pca_stats', [])
        else:
            self._preprocess_all_data(data_dict_list)
            if save_path is not None:
                print(f"Saving preprocessed data to {save_path}...")
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'processed_trials': self.processed_trials,
                        'win_labels': self.win_labels,
                        'trial_lengths': self.trial_lengths,
                        'pca_stats': self.pca_stats
                    }, f)

    def _apply_pca_to_mouse(self, calcium_data, n_components=None):
        """Apply PCA to a single mouse's calcium data"""
        # calcium_data shape: N * T (neurons x time)
        # Transpose to T * N for PCA
        data_transposed = calcium_data.T

        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_transposed)

        # Determine number of components
        if n_components is None:
            n_components = min(self.pca_dim, data_scaled.shape[1])
        else:
            n_components = min(n_components, data_scaled.shape[1])

        # Apply PCA
        pca_model = PCA(n_components=n_components)
        data_pca = pca_model.fit_transform(data_scaled)  # T * pca_dim

        explained_var = pca_model.explained_variance_ratio_.sum()

        return data_pca, pca_model, scaler, explained_var

    def _preprocess_all_data(self, data_dict_list):
        """Process all data with integrated PCA"""
        print("Processing data with integrated PCA...")

        pattern = re.compile(r'm(\d+)-m(\d+)')

        for data_dict in tqdm(data_dict_list, desc="Processing data files"):
            for key, value in data_dict.items():
                match = pattern.search(key)
                if not match:
                    continue

                mouse_ids = [match.group(1), match.group(2)]

                # Get whole calcium data for this mouse pair
                calcium_whole_1 = value[f"calcium_whole_{mouse_ids[0]}"]  # N1 * T
                calcium_whole_2 = value[f"calcium_whole_{mouse_ids[1]}"]  # N2 * T

                # Apply PCA to each mouse
                mouse1_pca_data, pca1, scaler1, var1 = self._apply_pca_to_mouse(calcium_whole_1)
                mouse2_pca_data, pca2, scaler2, var2 = self._apply_pca_to_mouse(calcium_whole_2)

                print(f"Mouse pair {key}: PCA variance explained - Mouse1: {var1:.4f}, Mouse2: {var2:.4f}")

                # Store PCA statistics
                self.pca_stats.append({
                    'mouse_pair': key,
                    'mouse1_neurons': calcium_whole_1.shape[0],
                    'mouse2_neurons': calcium_whole_2.shape[0],
                    'mouse1_var_explained': var1,
                    'mouse2_var_explained': var2,
                    'total_timepoints': calcium_whole_1.shape[1]
                })

                # Process trials for this mouse pair
                for trial_id, trial_data in value.items():
                    if "trial" not in trial_id or "time_start_end" not in trial_data:
                        continue

                    # Get trial boundaries
                    start_frame = int(trial_data["time_start_end"][0][0])
                    end_frame = int(trial_data["time_start_end"][1][0])
                    trial_length = end_frame - start_frame

                    # Store original trial length
                    self.trial_lengths.append(trial_length)

                    # Extract trial segments from PCA data
                    trial_segment_1 = mouse1_pca_data[start_frame:end_frame]
                    trial_segment_2 = mouse2_pca_data[start_frame:end_frame]

                    # Get winner label
                    winner = trial_data["winner"][-1]
                    label = 1 if mouse_ids[0] == winner else 0

                    # Store trial data (combined into one stream)
                    self.processed_trials.append({
                        'combined_data': np.concatenate([trial_segment_1, trial_segment_2], axis=0),
                        # Concatenate along time
                        'mouse_ids': mouse_ids,
                        'trial_length': trial_length,
                        'mouse_pair': key,
                        'mouse1_length': trial_segment_1.shape[0],
                        'mouse2_length': trial_segment_2.shape[0]
                    })
                    self.win_labels.append(label)

    def _augment_sequence(self, data, mask):
        """Apply data augmentation to a sequence"""
        # data shape: seq_length * pca_dim
        # mask shape: seq_length

        if not self.training:
            return data, mask

        # Channel shuffle
        if random.random() < self.channel_shuffle_prob:
            n_channels = int(self.channel_shuffle_ratio * data.shape[1])
            if n_channels > 0:
                shuffle_idx = np.random.choice(data.shape[1], n_channels, replace=False)
                shuffled_order = np.random.permutation(n_channels)
                data[:, shuffle_idx] = data[:, shuffle_idx[shuffled_order]]

        # Channel dropout
        if random.random() < self.channel_dropout_prob:
            n_dropout = int(self.channel_dropout_ratio * data.shape[1])
            if n_dropout > 0:
                dropout_idx = np.random.choice(data.shape[1], n_dropout, replace=False)
                data[:, dropout_idx] = 0

        # Time masking (only on valid positions)
        if random.random() < self.time_mask_prob:
            valid_positions = np.where(mask > 0)[0]  # Positions that are not padding
            if len(valid_positions) > 0:
                mask_length = int(self.time_mask_ratio * len(valid_positions))
                if mask_length > 0 and len(valid_positions) > mask_length:
                    mask_start_idx = np.random.randint(0, len(valid_positions) - mask_length)
                    mask_positions = valid_positions[mask_start_idx:mask_start_idx + mask_length]
                    data[mask_positions] = -1

        return data, mask

    def _prepare_sequence(self, trial_data, mouse1_length, mouse2_length):
        """Prepare sequence with proper length and padding, including masks for mouse identification"""
        original_length = trial_data.shape[0]

        # Split data back into mouse1 and mouse2
        mouse1_data = trial_data[:mouse1_length]
        mouse2_data = trial_data[mouse1_length:]

        if original_length > self.seq_length:
            # Sample seq_length//2 for each mouse from the same relative position
            half_seq = self.seq_length // 2

            if self.training and self.random_crop:
                # Random crop - same position for both mice
                if mouse1_length >= half_seq:
                    start_idx1 = np.random.randint(0, mouse1_length - half_seq + 1)
                    mouse1_sampled = mouse1_data[start_idx1:start_idx1 + half_seq]
                else:
                    mouse1_sampled = mouse1_data

                if mouse2_length >= half_seq:
                    start_idx2 = np.random.randint(0, mouse2_length - half_seq + 1)
                    mouse2_sampled = mouse2_data[start_idx2:start_idx2 + half_seq]
                else:
                    mouse2_sampled = mouse2_data
            else:
                # Take last segment during validation
                mouse1_sampled = mouse1_data[-half_seq:] if mouse1_length >= half_seq else mouse1_data
                mouse2_sampled = mouse2_data[-half_seq:] if mouse2_length >= half_seq else mouse2_data

            # Combine sampled data
            trial_final = np.vstack([mouse1_sampled, mouse2_sampled])
            mouse_mask = np.concatenate([
                np.ones(mouse1_sampled.shape[0], dtype=int),
                np.full(mouse2_sampled.shape[0], 2, dtype=int)
            ])

            # Pad if needed
            current_length = trial_final.shape[0]
            if current_length < self.seq_length:
                padding_length = self.seq_length - current_length
                padding_data = np.full((padding_length, trial_data.shape[1]), -1.0)
                trial_final = np.vstack([trial_final, padding_data])
                mouse_mask = np.concatenate([mouse_mask, np.zeros(padding_length, dtype=int)])

        elif original_length < self.seq_length:
            # Pad sequence
            padding_length = self.seq_length - original_length
            padding_data = np.full((padding_length, trial_data.shape[1]), -1.0)
            trial_final = np.vstack([trial_data, padding_data])

            # Create mouse mask
            mouse_mask = np.concatenate([
                np.ones(mouse1_length, dtype=int),
                np.full(mouse2_length, 2, dtype=int),
                np.zeros(padding_length, dtype=int)
            ])
        else:
            trial_final = trial_data
            mouse_mask = np.concatenate([
                np.ones(mouse1_length, dtype=int),
                np.full(mouse2_length, 2, dtype=int)
            ])

        return trial_final, mouse_mask

    def __len__(self):
        return len(self.processed_trials)

    def __getitem__(self, idx):
        trial = self.processed_trials[idx]

        # Prepare combined sequence
        combined_data, mouse_mask = self._prepare_sequence(
            trial['combined_data'].copy(),
            trial['mouse1_length'],
            trial['mouse2_length']
        )

        # Apply augmentation
        combined_data, mouse_mask = self._augment_sequence(combined_data, mouse_mask)

        # Convert to tensors
        combined_data = torch.FloatTensor(combined_data)

        # Add CLS token to mask only (value 3 for CLS token)
        cls_mask = torch.LongTensor([1])  # 3 = CLS token
        mouse_mask = torch.cat([cls_mask, torch.LongTensor(mouse_mask)], dim=0)

        # Get label
        label = torch.LongTensor([self.win_labels[idx]])[0]

        # # Random swap during training (swap mouse identities and their data)
        # if self.training and random.random() < 0.5:
        #     # Find positions for each mouse (skip CLS token at position 0)
        #     mouse1_positions = torch.where(mouse_mask[1:] == 1)[0] + 1  # +1 to account for CLS token
        #     mouse2_positions = torch.where(mouse_mask[1:] == 2)[0] + 1  # +1 to account for CLS token
        #
        #     # Swap the data at these positions
        #     if len(mouse1_positions) > 0 and len(mouse2_positions) > 0:
        #         # Store mouse1 data temporarily
        #         mouse1_data = combined_data[mouse1_positions].clone()
        #         mouse2_data = combined_data[mouse2_positions].clone()
        #
        #         # Swap data
        #         combined_data[mouse1_positions] = mouse2_data
        #         combined_data[mouse2_positions] = mouse1_data
        #
        #         # Swap mouse identities in the mask
        #         mouse_mask[mouse1_positions] = 2
        #         mouse_mask[mouse2_positions] = 1
        #
        #         # Flip label
        #         label = 1 - label

        return combined_data, mouse_mask, label


class CalciumDataset_two_stream_pair_pca(Dataset):
    """Calcium imaging dataset with integrated PCA and data augmentation"""

    def __init__(self, data_dict_list, seq_length=196, pca_dim=128, save_path=None,
                 training=True,
                 # Augmentation parameters
                 random_crop=True,
                 channel_shuffle_prob=0.2,
                 channel_shuffle_ratio=0.1,
                 channel_dropout_prob=0.2,
                 channel_dropout_ratio=0.1,
                 time_mask_prob=0.2,
                 time_mask_ratio=0.1):

        self.seq_length = seq_length
        self.pca_dim = pca_dim
        self.save_path = save_path
        self.training = training

        # Augmentation settings
        self.random_crop = random_crop
        self.channel_shuffle_prob = channel_shuffle_prob
        self.channel_shuffle_ratio = channel_shuffle_ratio
        self.channel_dropout_prob = channel_dropout_prob
        self.channel_dropout_ratio = channel_dropout_ratio
        self.time_mask_prob = time_mask_prob
        self.time_mask_ratio = time_mask_ratio

        # Initialize storage
        self.processed_trials = []
        self.win_labels = []
        self.trial_lengths = []  # Store original trial lengths for analysis
        self.pca_stats = []  # Store PCA statistics

        # Check for cached data
        if save_path is not None and os.path.exists(save_path):
            print(f"Loading preprocessed data from {save_path}...")
            with open(save_path, 'rb') as f:
                cache = pickle.load(f)
                self.processed_trials = cache['processed_trials']
                self.win_labels = cache['win_labels']
                self.trial_lengths = cache.get('trial_lengths', [])
                self.pca_stats = cache.get('pca_stats', [])
        else:
            self._preprocess_all_data(data_dict_list)
            if save_path is not None:
                print(f"Saving preprocessed data to {save_path}...")
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'processed_trials': self.processed_trials,
                        'win_labels': self.win_labels,
                        'trial_lengths': self.trial_lengths,
                        'pca_stats': self.pca_stats
                    }, f)

    def _apply_pca_to_mouse(self, calcium_data, n_components=None):
        """Apply PCA to a single mouse's calcium data"""
        # calcium_data shape: N * T (neurons x time)
        # Transpose to T * N for PCA
        data_transposed = calcium_data.T

        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_transposed)

        # Determine number of components
        if n_components is None:
            n_components = min(self.pca_dim, data_scaled.shape[1])
        else:
            n_components = min(n_components, data_scaled.shape[1])

        # Apply PCA
        pca_model = PCA(n_components=n_components)
        data_pca = pca_model.fit_transform(data_scaled)  # T * pca_dim

        explained_var = pca_model.explained_variance_ratio_.sum()

        return data_pca, pca_model, scaler, explained_var

    def _preprocess_all_data(self, data_dict_list):
        """Process all data with integrated PCA"""
        print("Processing data with integrated PCA...")

        pattern = re.compile(r'm(\d+)-m(\d+)')

        for data_dict in tqdm(data_dict_list, desc="Processing data files"):
            for key, value in data_dict.items():
                match = pattern.search(key)
                if not match:
                    continue

                mouse_ids = [match.group(1), match.group(2)]

                # Get whole calcium data for this mouse pair
                calcium_whole_1 = value[f"calcium_whole_{mouse_ids[0]}"]  # N1 * T
                calcium_whole_2 = value[f"calcium_whole_{mouse_ids[1]}"]  # N2 * T

                # Apply PCA to each mouse
                mouse1_pca_data, pca1, scaler1, var1 = self._apply_pca_to_mouse(calcium_whole_1)
                mouse2_pca_data, pca2, scaler2, var2 = self._apply_pca_to_mouse(calcium_whole_2)

                print(f"Mouse pair {key}: PCA variance explained - Mouse1: {var1:.4f}, Mouse2: {var2:.4f}")

                # Store PCA statistics
                self.pca_stats.append({
                    'mouse_pair': key,
                    'mouse1_neurons': calcium_whole_1.shape[0],
                    'mouse2_neurons': calcium_whole_2.shape[0],
                    'mouse1_var_explained': var1,
                    'mouse2_var_explained': var2,
                    'total_timepoints': calcium_whole_1.shape[1]
                })

                # Process trials for this mouse pair
                for trial_id, trial_data in value.items():
                    if "trial" not in trial_id or "time_start_end" not in trial_data:
                        continue

                    # Get trial boundaries
                    start_frame = int(trial_data["time_start_end"][0][0])
                    end_frame = int(trial_data["time_start_end"][1][0])
                    trial_length = end_frame - start_frame

                    # Store original trial length
                    self.trial_lengths.append(trial_length)

                    # Extract trial segments from PCA data
                    trial_segment_1 = mouse1_pca_data[start_frame:end_frame]
                    trial_segment_2 = mouse2_pca_data[start_frame:end_frame]

                    # Get winner label
                    winner = trial_data["winner"][-1]
                    label = 1 if mouse_ids[0] == winner else 0

                    # Store trial data
                    self.processed_trials.append({
                        'mouse1_data': trial_segment_1,
                        'mouse2_data': trial_segment_2,
                        'mouse_ids': mouse_ids,
                        'trial_length': trial_length,
                        'mouse_pair': key
                    })
                    self.win_labels.append(label)

    def _augment_sequence(self, data, mask):
        """Apply data augmentation to a sequence"""
        # data shape: seq_length * pca_dim
        # mask shape: seq_length

        if not self.training:
            return data, mask

        # Channel shuffle
        if random.random() < self.channel_shuffle_prob:
            n_channels = int(self.channel_shuffle_ratio * data.shape[1])
            if n_channels > 0:
                shuffle_idx = np.random.choice(data.shape[1], n_channels, replace=False)
                shuffled_order = np.random.permutation(n_channels)
                data[:, shuffle_idx] = data[:, shuffle_idx[shuffled_order]]

        # Channel dropout
        if random.random() < self.channel_dropout_prob:
            n_dropout = int(self.channel_dropout_ratio * data.shape[1])
            if n_dropout > 0:
                dropout_idx = np.random.choice(data.shape[1], n_dropout, replace=False)
                data[:, dropout_idx] = 0

        # Time masking
        if random.random() < self.time_mask_prob:
            valid_length = mask.sum()
            mask_length = int(self.time_mask_ratio * valid_length)
            if mask_length > 0 and valid_length > mask_length:
                mask_start = np.random.randint(0, valid_length - mask_length)
                data[mask_start:mask_start + mask_length] = -1

        return data, mask

    def _prepare_sequence(self, trial_data):
        """Prepare sequence with proper length and padding"""
        original_length = trial_data.shape[0]

        if original_length > self.seq_length:
            if self.training and self.random_crop:
                # Random crop during training
                start_idx = np.random.randint(0, original_length - self.seq_length + 1)
                trial_final = trial_data[start_idx:start_idx + self.seq_length]
            else:
                # Take last segment during validation
                trial_final = trial_data[-self.seq_length:]
            mask = np.ones(self.seq_length, dtype=bool)
        elif original_length < self.seq_length:
            # Pad sequence
            padding_length = self.seq_length - original_length
            padding = np.full((padding_length, trial_data.shape[1]), -1.0)
            trial_final = np.vstack([trial_data, padding])
            mask = np.concatenate([
                np.ones(original_length, dtype=bool),
                np.zeros(padding_length, dtype=bool)
            ])
        else:
            trial_final = trial_data
            mask = np.ones(self.seq_length, dtype=bool)

        return trial_final, mask

    def __len__(self):
        return len(self.processed_trials)

    def __getitem__(self, idx):
        trial = self.processed_trials[idx]

        # Prepare sequences
        mouse1_data, mouse1_mask = self._prepare_sequence(trial['mouse1_data'].copy())
        mouse2_data, mouse2_mask = self._prepare_sequence(trial['mouse2_data'].copy())

        # Apply augmentation
        mouse1_data, mouse1_mask = self._augment_sequence(mouse1_data, mouse1_mask)
        mouse2_data, mouse2_mask = self._augment_sequence(mouse2_data, mouse2_mask)

        # Convert to tensors
        mouse1_data = torch.FloatTensor(mouse1_data)
        mouse2_data = torch.FloatTensor(mouse2_data)
        mouse1_mask = torch.BoolTensor(mouse1_mask)
        mouse2_mask = torch.BoolTensor(mouse2_mask)

        # Get label
        label = torch.LongTensor([self.win_labels[idx]])[0]

        # Random swap during training
        if self.training and random.random() < 0.5:
            mouse1_data, mouse2_data = mouse2_data, mouse1_data
            mouse1_mask, mouse2_mask = mouse2_mask, mouse1_mask
            label = 1 - label

        return mouse1_data, mouse2_data, mouse1_mask, mouse2_mask, label

    def get_sample_data(self, n_samples=5):
        """Get sample data for visualization without augmentation"""
        samples = []
        for i in range(min(n_samples, len(self.processed_trials))):
            trial = self.processed_trials[i]
            samples.append({
                'mouse1_data': trial['mouse1_data'],
                'mouse2_data': trial['mouse2_data'],
                'label': self.win_labels[i],
                'trial_length': trial['trial_length'],
                'mouse_pair': trial['mouse_pair']
            })
        return samples