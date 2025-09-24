import os
import os.path
import pickle
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from engine import train_model
from datasets import CalciumDataset_one_stream_pair_pca
from model import Neuron_BERT
from datasets import (
    split_data_mouse_level,
    split_data_trial_level,
    split_data_file_level,
    analyze_split_distribution,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train single-stream Neuron BERT model")

    # Data & splitting
    parser.add_argument("--data_dir", type=str, default=r"D:\BBNC\PICO\code\PICO_figure_plot\tube_test\data",
                        help="Directory containing pickled data dictionaries")
    parser.add_argument("--split_strategy", type=str, default="trial_level",
                        choices=["mouse_level", "trial_level", "file_level"], help="Dataset split strategy")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.0, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")

    # Model hyperparameters
    parser.add_argument("--input_dim", type=int, default=128, help="Input PCA dimension")
    parser.add_argument("--seq_length", type=int, default=392, help="Sequence length")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP expansion ratio")
    parser.add_argument("--drop_ratio", type=float, default=0.3, help="Dropout ratio")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")

    # Augmentation parameters
    parser.add_argument("--no_random_crop", action="store_true", help="Disable random crop during training")
    parser.add_argument("--channel_shuffle_prob", type=float, default=0.8, help="Prob. of channel shuffle")
    parser.add_argument("--channel_shuffle_ratio", type=float, default=0.3, help="Ratio of channels to shuffle")
    parser.add_argument("--channel_dropout_prob", type=float, default=0.2, help="Prob. of channel dropout")
    parser.add_argument("--channel_dropout_ratio", type=float, default=0.1, help="Ratio of channels to drop")
    parser.add_argument("--time_mask_prob", type=float, default=0.2, help="Probability of time masking")
    parser.add_argument("--time_mask_ratio", type=float, default=0.1, help="Ratio of time steps to mask")

    # Logging / run mgmt
    parser.add_argument("--runs_root", type=str, default="runs/bert_tube_test", help="Root directory for runs")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name (otherwise auto-generated)")
    parser.add_argument("--print_example_batch", action="store_true", help="Print an example batch's shapes")
    parser.add_argument("--disable_cache", action="store_true", help="Disable cached preprocessed data usage")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Collect data
    data_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
                  if os.path.isfile(os.path.join(args.data_dir, f))]
    if len(data_paths) == 0:
        raise FileNotFoundError(f"No data files found in directory: {args.data_dir}")

    data_dict_list = []
    for data_path in data_paths:
        with open(data_path, "rb") as f:
            data_dict_list.append(pickle.load(f))

    print(f"Loaded {len(data_dict_list)} data files from {args.data_dir}")

    # Split
    if args.split_strategy == "mouse_level":
        print("\n=== Using Mouse Level Split ===")
        train_data_list, val_data_list, test_data_list = split_data_mouse_level(
            data_dict_list,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
        )
    elif args.split_strategy == "trial_level":
        print("\n=== Using Trial Level Split ===")
        train_data_list, val_data_list, test_data_list = split_data_trial_level(
            data_dict_list,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
        )
    elif args.split_strategy == "file_level":
        print("\n=== Using File Level Split ===")
        train_data_list, val_data_list, test_data_list = split_data_file_level(
            data_dict_list,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown split strategy: {args.split_strategy}")

    # Analyze splits
    print("\nSplit Analysis:")
    analyze_split_distribution(train_data_list, "Training")
    analyze_split_distribution(val_data_list, "Validation")
    if args.test_ratio > 0:
        analyze_split_distribution(test_data_list, "Test")

    # Cache paths (optional)
    train_cache_path = None if args.disable_cache else os.path.join("cache", f"train_data_bert_{args.split_strategy}_pca.pkl")
    val_cache_path = None if args.disable_cache else os.path.join("cache", f"val_data_bert_{args.split_strategy}_pca.pkl")

    # Datasets
    train_dataset = CalciumDataset_one_stream_pair_pca(
        train_data_list,
        seq_length=args.seq_length,
        pca_dim=args.input_dim,
        save_path=train_cache_path,
        training=True,
        random_crop=not args.no_random_crop,
        channel_shuffle_prob=args.channel_shuffle_prob,
        channel_shuffle_ratio=args.channel_shuffle_ratio,
        channel_dropout_prob=args.channel_dropout_prob,
        channel_dropout_ratio=args.channel_dropout_ratio,
        time_mask_prob=args.time_mask_prob,
        time_mask_ratio=args.time_mask_ratio,
    )
    val_dataset = CalciumDataset_one_stream_pair_pca(
        val_data_list,
        seq_length=args.seq_length,
        pca_dim=args.input_dim,
        save_path=val_cache_path,
        training=False,
        random_crop=False,
        channel_shuffle_prob=0.0,
        channel_dropout_prob=0.0,
        time_mask_prob=0.0,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("\nDataset sizes:")
    print(f"Training:   {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"PCA dim:    {args.input_dim}")
    print(f"Seq length: {args.seq_length}")

    if args.print_example_batch:
        first_batch = next(iter(train_loader))
        mouse_data, mouse_mask, labels = first_batch
        print("\nExample batch:")
        print(f"mouse_data shape: {mouse_data.shape}")
        print(f"mouse_mask shape: {mouse_mask.shape}")
        print(f"labels: {labels}")

    # Run directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.run_name is None:
        folder_name = (
            f"{args.split_strategy}_emb{args.embed_dim}_d{args.depth}_mlp{int(args.mlp_ratio)}_"
            f"drop{int(args.drop_ratio * 100)}_lr{args.lr}_" + timestamp
        )
    else:
        folder_name = args.run_name
    save_dir = os.path.join(args.runs_root, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving checkpoints & logs to: {save_dir}")

    # Model
    model = Neuron_BERT(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        seq_length=args.seq_length,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_ratio=args.drop_ratio,
    )

    print(model)

    best_metrics = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
    )

    print(
        f"Best validation metrics: Acc={best_metrics['accuracy']:.2f}% P={best_metrics['precision']:.2f}% "
        f"R={best_metrics['recall']:.2f}% F1={best_metrics['f1_score']:.2f}%"
    )


if __name__ == "__main__":
    main()