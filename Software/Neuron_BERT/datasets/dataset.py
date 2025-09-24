import os.path
import pickle
import re

import matplotlib.pyplot as plt
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

from tqdm import tqdm

class CalciumDataset(Dataset):
    def __init__(self, calcium_data, win_label, seq_length=196, pca_dim=128):
        self.calcium_data = calcium_data
        self.win_label = win_label
        self.seq_length = seq_length
        self.pca_dim = pca_dim

        self.processed_data = []
        self.masks = []
        self._preprocess_data()

    def _preprocess_data(self):
        for trial_i in tqdm(range(len(self.calcium_data[0]))):

            segment_lengths = []
            pca_segments = []

            for mouse_idx in range(2):
                trial_data = self.calcium_data[mouse_idx][trial_i].T  # (T, N)
                scaler = StandardScaler()
                trial_data = scaler.fit_transform(trial_data)

                pca_model = PCA(n_components=self.pca_dim)
                if trial_data.shape[0] >= self.pca_dim:
                    trial_processed = pca_model.fit_transform(trial_data)
                else:
                    padded = np.concatenate(
                        [trial_data, np.zeros((self.pca_dim - trial_data.shape[0], trial_data.shape[1]))],
                        axis=0
                    )
                    trial_processed = pca_model.fit_transform(padded)[:trial_data.shape[0]]

                pca_segments.append(trial_processed)
                segment_lengths.append(trial_processed.shape[0])

            half_seq = self.seq_length // 2
            mouse1_len, mouse2_len = segment_lengths

            if mouse1_len >= half_seq and mouse2_len >= half_seq:
                mouse1_data = pca_segments[0][-half_seq:]
                mouse2_data = pca_segments[1][-half_seq:]
            else:
                # Assign as much as possible to mouse1, then mouse2
                remaining = self.seq_length
                if mouse1_len <= half_seq:
                    mouse1_data = pca_segments[0]
                    remaining -= mouse1_data.shape[0]
                    mouse2_data = pca_segments[1][-remaining:] if remaining < segment_lengths[1] else pca_segments[1]
                else:
                    mouse2_data = pca_segments[1]
                    remaining -= mouse2_data.shape[0]
                    mouse1_data = pca_segments[0][-remaining:] if remaining < segment_lengths[0] else pca_segments[0]

            # Concatenate and mask
            data_parts = [mouse1_data, mouse2_data]
            mask_parts = [
                np.full(mouse1_data.shape[0]+1, 1, dtype=int), # label for sum with cls token part
                np.full(mouse2_data.shape[0], 2, dtype=int),
            ]

            current_len = mouse1_data.shape[0] + mouse2_data.shape[0]
            if current_len < self.seq_length:
                pad_len = self.seq_length - current_len
                data_parts.append(np.full((pad_len, self.pca_dim), -1.0))
                mask_parts.append(np.zeros(pad_len, dtype=int))

            full_data = np.vstack(data_parts)
            full_mask = np.concatenate(mask_parts)

            self.processed_data.append(full_data)
            self.masks.append(full_mask)

    def __len__(self):
        return len(self.calcium_data[0])

    def __getitem__(self, idx):
        data = torch.FloatTensor(self.processed_data[idx])  # (seq_length, pca_dim)
        mask = torch.LongTensor(self.masks[idx])            # (seq_length,)
        label = torch.LongTensor([self.win_label[0][idx]])[0]
        return data, mask, label


class CalciumDataset_two_stream(Dataset):
    """钙成像数据集类 - 固定长度196"""

    def __init__(self, calcium_data, win_label, seq_length=196, pca_dim=128, save_path=None):
        self.calcium_data = calcium_data
        self.win_label = win_label
        self.seq_length = seq_length
        self.pca_dim = pca_dim
        self.save_path = save_path

        self.processed_data = [[], []]
        self.masks = [[], []]

        if save_path is not None and os.path.exists(save_path):
            print(f"Loading preprocessed data from {save_path}...")
            with open(save_path, 'rb') as f:
                cache = pickle.load(f)
                self.processed_data = cache['processed_data']
                self.masks = cache['masks']
        else:
            self._preprocess_data()
            if save_path is not None:
                print(f"Saving preprocessed data to {save_path}...")
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'processed_data': self.processed_data,
                        'masks': self.masks
                    }, f)

    def _preprocess_data(self):
        for trial_i in tqdm(range(len(self.calcium_data[0])), desc="Preprocessing data"):
            for mouse_idx in range(2):
                trial_data = self.calcium_data[mouse_idx][trial_i].T  # N, T -> T, N
                scaler = StandardScaler()
                trial_data = scaler.fit_transform(trial_data)

                pca_model = PCA(n_components=self.pca_dim)
                if trial_data.shape[0] >= self.pca_dim:
                    trial_processed = pca_model.fit_transform(trial_data)
                else:
                    padding_rows = self.pca_dim - trial_data.shape[0]
                    trial_padded = np.concatenate([trial_data, np.zeros((padding_rows, trial_data.shape[1]))], axis=0)
                    trial_processed = pca_model.fit_transform(trial_padded)[:trial_data.shape[0]]

                original_length = trial_data.shape[0]

                if original_length > self.seq_length:
                    trial_final = trial_processed[-self.seq_length:]
                    mask = np.ones(self.seq_length, dtype=bool)
                elif original_length < self.seq_length:
                    padding_length = self.seq_length - original_length
                    padding = np.full((padding_length, trial_processed.shape[1]), -1.0)
                    trial_final = np.vstack([trial_processed, padding])
                    mask = np.concatenate([
                        np.ones(original_length, dtype=bool),
                        np.zeros(padding_length, dtype=bool)
                    ])
                else:
                    trial_final = trial_processed
                    mask = np.ones(self.seq_length, dtype=bool)

                self.processed_data[mouse_idx].append(trial_final)
                self.masks[mouse_idx].append(mask)

    def __len__(self):
        return len(self.calcium_data[0])

    def __getitem__(self, idx):
        mouse1_data = torch.FloatTensor(self.processed_data[0][idx])
        mouse2_data = torch.FloatTensor(self.processed_data[1][idx])
        mouse1_mask = torch.BoolTensor(self.masks[0][idx])
        mouse2_mask = torch.BoolTensor(self.masks[1][idx])
        label = torch.LongTensor([self.win_label[0][idx]])[0]
        return mouse1_data, mouse2_data, mouse1_mask, mouse2_mask, label



if __name__ == "__main__":


    data_dir = r"D:\BBNC\PICO\code\PICO_figure_plot\tube_test\data"
    data_paths = [os.path.join(data_dir, tmp) for tmp in os.listdir(data_dir)]

    calcium_data = [[], []]
    win_label = [[], []]

    for data_path in data_paths:

        pattern = re.compile(r'm(\d+)-m(\d+)')

        with open(data_path, "rb") as f:
            data_dir = pickle.load(f)

        for key, value in data_dir.items():

            match = pattern.search(key)
            mouse_id = [match.group(1), match.group(2)]
            for trial_id, data in value.items():
                if "trial" not in trial_id:
                    continue

                for i, id in enumerate(mouse_id):
                    calcium_data[i].append(data[f"calcium_{id}"])  # Neurons * T
                    win_label[i].append(1 if id == data["winner"][-1] else 0)

    dataset = CalciumDataset(calcium_data, win_label, seq_length=196, pca_dim=128)

    data, mask, label = dataset[0]

    fig, axs = plt.subplots(1, 2, figsize=(5, 3))

    # N * T (T1, T2, padding)

    axs[0].plot(mask.numpy())

    axs[1].imshow(data.numpy().T, cmap="viridis", aspect="auto")
    axs[1].set_xlabel("Time (frames)")
    axs[1].set_ylabel("neurons")

    plt.show()







