import os
import pickle
import random
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np


def split_data_file_level(data_dict_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split data at file level - entire files go to different splits
    This ensures complete separation of experimental sessions/batches

    Args:
        data_dict_list: List of data dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        train_data_list, val_data_list, test_data_list
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(random_seed)
    np.random.seed(random_seed)

    n_files = len(data_dict_list)
    file_indices = list(range(n_files))

    print(f"Found {n_files} data files")

    # Split file indices
    random.shuffle(file_indices)

    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    n_test = n_files - n_train - n_val

    train_file_indices = set(file_indices[:n_train])
    val_file_indices = set(file_indices[n_train:n_train + n_val])
    test_file_indices = set(file_indices[n_train + n_val:])

    print(f"Train files ({len(train_file_indices)}): {sorted(train_file_indices)}")
    print(f"Val files ({len(val_file_indices)}): {sorted(val_file_indices)}")
    print(f"Test files ({len(test_file_indices)}): {sorted(test_file_indices)}")

    # Split data lists based on file indices
    train_data_list = [data_dict_list[i] for i in sorted(train_file_indices)]
    val_data_list = [data_dict_list[i] for i in sorted(val_file_indices)]
    test_data_list = [data_dict_list[i] for i in sorted(test_file_indices)]

    return train_data_list, val_data_list, test_data_list


def split_data_mouse_level(data_dict_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split data at mouse level - ensures same mouse pairs don't appear in different splits
    Each data_dict file is treated as a unique session/experiment, so mouse pairs are
    identified by (file_index, mouse_pair_name) to avoid conflicts between files.

    Args:
        data_dict_list: List of data dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        train_data_list, val_data_list, test_data_list
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(random_seed)
    np.random.seed(random_seed)

    # Extract all unique mouse pairs with file context
    mouse_pair_file_combinations = []
    pattern = re.compile(r'm(\d+)-m(\d+)')

    for file_idx, data_dict in enumerate(data_dict_list):
        for key in data_dict.keys():
            match = pattern.search(key)
            if match:
                # Create unique identifier: (file_index, mouse_pair_name)
                unique_pair_id = (file_idx, key)
                mouse_pair_file_combinations.append(unique_pair_id)

    print(f"Found {len(mouse_pair_file_combinations)} unique mouse pair-file combinations:")
    for file_idx, pair_name in mouse_pair_file_combinations:
        print(f"  File {file_idx}: {pair_name}")

    # Split mouse pair-file combinations
    random.shuffle(mouse_pair_file_combinations)

    n_total = len(mouse_pair_file_combinations)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_combinations = set(mouse_pair_file_combinations[:n_train])
    val_combinations = set(mouse_pair_file_combinations[n_train:n_train + n_val])
    test_combinations = set(mouse_pair_file_combinations[n_train + n_val:])

    print(f"\nTrain combinations ({len(train_combinations)}):")
    for file_idx, pair_name in train_combinations:
        print(f"  File {file_idx}: {pair_name}")

    print(f"\nVal combinations ({len(val_combinations)}):")
    for file_idx, pair_name in val_combinations:
        print(f"  File {file_idx}: {pair_name}")

    print(f"\nTest combinations ({len(test_combinations)}):")
    for file_idx, pair_name in test_combinations:
        print(f"  File {file_idx}: {pair_name}")

    # Split data dictionaries based on mouse pair-file combinations
    train_data_list = []
    val_data_list = []
    test_data_list = []

    for file_idx, data_dict in enumerate(data_dict_list):
        train_dict = {}
        val_dict = {}
        test_dict = {}

        for key, value in data_dict.items():
            match = pattern.search(key)
            if match:
                combination_id = (file_idx, key)
                if combination_id in train_combinations:
                    train_dict[key] = value
                elif combination_id in val_combinations:
                    val_dict[key] = value
                elif combination_id in test_combinations:
                    test_dict[key] = value

        if train_dict:
            train_data_list.append(train_dict)
        if val_dict:
            val_data_list.append(val_dict)
        if test_dict:
            test_data_list.append(test_dict)

    return train_data_list, val_data_list, test_data_list


def split_data_trial_level(data_dict_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split data at trial level - same mouse pairs can appear in different splits but different trials

    Args:
        data_dict_list: List of data dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        train_data_list, val_data_list, test_data_list
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(random_seed)
    np.random.seed(random_seed)

    # Collect all trials with their identifiers
    all_trials = []
    pattern = re.compile(r'm(\d+)-m(\d+)')

    for data_dict_idx, data_dict in enumerate(data_dict_list):
        for mouse_pair_key, mouse_pair_data in data_dict.items():
            match = pattern.search(mouse_pair_key)
            if not match:
                continue

            # Find all trials for this mouse pair
            for trial_key, trial_data in mouse_pair_data.items():
                if "trial" in trial_key and isinstance(trial_data, dict):
                    all_trials.append({
                        'data_dict_idx': data_dict_idx,
                        'mouse_pair_key': mouse_pair_key,
                        'trial_key': trial_key,
                        'trial_data': trial_data
                    })

    print(f"Found {len(all_trials)} total trials")

    # Shuffle and split trials
    random.shuffle(all_trials)

    n_total = len(all_trials)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_trials = all_trials[:n_train]
    val_trials = all_trials[n_train:n_train + n_val]
    test_trials = all_trials[n_train + n_val:]

    print(f"Train trials: {len(train_trials)}")
    print(f"Val trials: {len(val_trials)}")
    print(f"Test trials: {len(test_trials)}")

    # Build new data dictionaries for each split
    def build_split_data(trial_list):
        # Group trials by data_dict_idx and mouse_pair
        split_structure = defaultdict(lambda: defaultdict(dict))

        for trial_info in trial_list:
            data_idx = trial_info['data_dict_idx']
            mouse_pair = trial_info['mouse_pair_key']
            trial_key = trial_info['trial_key']
            trial_data = trial_info['trial_data']

            # Copy mouse pair metadata if not already present
            original_data_dict = data_dict_list[data_idx]
            if mouse_pair not in split_structure[data_idx]:
                # Copy all non-trial data (calcium_whole, neuron_center, etc.)
                for key, value in original_data_dict[mouse_pair].items():
                    if "trial" not in key:
                        split_structure[data_idx][mouse_pair][key] = value

            # Add this specific trial
            split_structure[data_idx][mouse_pair][trial_key] = trial_data

        # Convert to list format
        split_data_list = []
        for data_idx in sorted(split_structure.keys()):
            split_data_list.append(dict(split_structure[data_idx]))

        return split_data_list

    train_data_list = build_split_data(train_trials)
    val_data_list = build_split_data(val_trials)
    test_data_list = build_split_data(test_trials)

    return train_data_list, val_data_list, test_data_list


def analyze_split_distribution(data_list, split_name="Split"):
    """Analyze the distribution of mouse pairs and trials in a data split"""
    mouse_pair_file_combinations = []
    trial_count = 0
    pattern = re.compile(r'm(\d+)-m(\d+)')

    for file_idx, data_dict in enumerate(data_list):
        for key, value in data_dict.items():
            match = pattern.search(key)
            if match:
                mouse_pair_file_combinations.append(f"File{file_idx}:{key}")
                # Count trials
                for trial_key in value.keys():
                    if "trial" in trial_key:
                        trial_count += 1

    print(f"{split_name}:")
    print(f"  - Mouse pair-file combinations: {len(mouse_pair_file_combinations)}")
    for combination in mouse_pair_file_combinations:
        print(f"    {combination}")
    print(f"  - Total trials: {trial_count}")
    print()


def data_spliting_test():
    """Example of how to use the splitting functions"""

    # Load your data (replace with your actual loading code)
    data_dir = r"D:\BBNC\PICO\code\PICO_figure_plot\tube_test\data"
    data_paths = [os.path.join(data_dir, tmp) for tmp in os.listdir(data_dir)]

    data_dict_list = []
    for data_path in data_paths:
        with open(data_path, "rb") as f:
            data_dict = pickle.load(f)
            data_dict_list.append(data_dict)

    print("=== MOUSE LEVEL SPLIT ===")
    print("Each mouse pair appears in only one split (train/val/test)")
    print("Good for: Generalization to new mouse pairs")
    print()

    train_mouse, val_mouse, test_mouse = split_data_mouse_level(
        data_dict_list,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    analyze_split_distribution(train_mouse, "Training Set")
    analyze_split_distribution(val_mouse, "Validation Set")
    analyze_split_distribution(test_mouse, "Test Set")

    print("\n" + "=" * 50 + "\n")

    print("=== TRIAL LEVEL SPLIT ===")
    print("Same mouse pairs can appear in different splits, but with different trials")
    print("Good for: Learning trial-specific patterns, larger training set")
    print()

    train_trial, val_trial, test_trial = split_data_trial_level(
        data_dict_list,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    analyze_split_distribution(train_trial, "Training Set")
    analyze_split_distribution(val_trial, "Validation Set")
    analyze_split_distribution(test_trial, "Test Set")

    return {
        'mouse_level': (train_mouse, val_mouse, test_mouse),
        'trial_level': (train_trial, val_trial, test_trial)
    }



if __name__ == "__main__":

    data_spliting_test()




