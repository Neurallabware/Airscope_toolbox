import os
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import seaborn as sns


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_confusion_matrix(y_true, y_pred, save_path, epoch=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Mouse 2 Wins', 'Mouse 1 Wins'],
                yticklabels=['Mouse 2 Wins', 'Mouse 1 Wins'])
    plt.title(f'Confusion Matrix{f" - Epoch {epoch}" if epoch else ""}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    if epoch:
        plt.savefig(os.path.join(save_path, f'confusion_matrix_epoch_{epoch}.pdf'))
    else:
        plt.savefig(os.path.join(save_path, 'final_confusion_matrix.pdf'))
    plt.close()


def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, save_dir="./"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    best_epoch = 0

    # Prepare log file
    log_path = os.path.join(save_dir, 'log.txt')
    with open(log_path, 'w') as log_file:
        log_file.write(f"Model architecture:\n{model}\n\n")

    # Lists to store metrics
    train_losses, val_losses = [], []
    train_metrics_history = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    val_metrics_history = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    lrs = []

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        running_loss, all_preds, all_labels = 0.0, [], []

        for mouse_data, mouse_mask, labels in train_loader:
            mouse_data = mouse_data.to(device)
            mouse_mask = mouse_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(mouse_data, mouse_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(all_labels)
        train_metrics = calculate_metrics(all_labels, all_preds)

        # Validation
        model.eval()
        val_running_loss, val_all_preds, val_all_labels = 0.0, [], []

        with torch.no_grad():
            for mouse_data, mouse_mask, labels in val_loader:
                mouse_data = mouse_data.to(device)
                mouse_mask = mouse_mask.to(device)
                labels = labels.to(device)

                outputs = model(mouse_data, mouse_mask)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_all_labels)
        val_metrics = calculate_metrics(val_all_labels, val_all_preds)

        # Scheduler and LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(current_lr)

        for metric in train_metrics.keys():
            train_metrics_history[metric].append(train_metrics[metric] * 100)
            val_metrics_history[metric].append(val_metrics[metric] * 100)

        # Save best model based on accuracy
        if val_metrics['accuracy'] > best_val_metrics['accuracy']:
            best_val_metrics = val_metrics.copy()
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

            # Save confusion matrix for best model
            plot_confusion_matrix(val_all_labels, val_all_preds, save_dir, epoch)

        # Log epoch
        log_line = (f"Epoch {epoch:03d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy'] * 100:.2f}%, "
                    f"P: {train_metrics['precision'] * 100:.2f}%, R: {train_metrics['recall'] * 100:.2f}%, "
                    f"F1: {train_metrics['f1_score'] * 100:.2f}% | "
                    f"Val Acc: {val_metrics['accuracy'] * 100:.2f}%, "
                    f"P: {val_metrics['precision'] * 100:.2f}%, R: {val_metrics['recall'] * 100:.2f}%, "
                    f"F1: {val_metrics['f1_score'] * 100:.2f}% | "
                    f"LR: {current_lr:.6f}\n")
        print(log_line.strip())
        with open(log_path, 'a') as log_file:
            log_file.write(log_line)

    # Plot training curves
    epochs = list(range(1, num_epochs + 1))
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Loss Curve
    axs[0, 0].plot(epochs, train_losses, label='Train Loss')
    axs[0, 0].set_title('Loss Curve')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Accuracy Curve
    axs[0, 1].plot(epochs, train_metrics_history['accuracy'], label='Train Acc')
    axs[0, 1].plot(epochs, val_metrics_history['accuracy'], label='Val Acc')
    axs[0, 1].set_title('Accuracy Curve')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy (%)')
    axs[0, 1].legend()

    # Precision Curve
    axs[0, 2].plot(epochs, train_metrics_history['precision'], label='Train Precision')
    axs[0, 2].plot(epochs, val_metrics_history['precision'], label='Val Precision')
    axs[0, 2].set_title('Precision Curve')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('Precision (%)')
    axs[0, 2].legend()

    # Recall Curve
    axs[1, 0].plot(epochs, train_metrics_history['recall'], label='Train Recall')
    axs[1, 0].plot(epochs, val_metrics_history['recall'], label='Val Recall')
    axs[1, 0].set_title('Recall Curve')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Recall (%)')
    axs[1, 0].legend()

    # F1 Score Curve
    axs[1, 1].plot(epochs, train_metrics_history['f1_score'], label='Train F1')
    axs[1, 1].plot(epochs, val_metrics_history['f1_score'], label='Val F1')
    axs[1, 1].set_title('F1 Score Curve')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('F1 Score (%)')
    axs[1, 1].legend()

    # Learning Rate Curve
    axs[1, 2].plot(epochs, lrs)
    axs[1, 2].set_title('LR Schedule')
    axs[1, 2].set_xlabel('Epoch')
    axs[1, 2].set_ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.pdf'))
    plt.close()

    # Convert best metrics to percentages for return
    for key in best_val_metrics:
        best_val_metrics[key] *= 100

    print(f"\nBest model saved at epoch {best_epoch}")
    print(f"Best validation metrics: Acc: {best_val_metrics['accuracy']:.2f}%, "
          f"P: {best_val_metrics['precision']:.2f}%, R: {best_val_metrics['recall']:.2f}%, "
          f"F1: {best_val_metrics['f1_score']:.2f}%")

    return best_val_metrics


def train_model_two_stream(model, train_loader, val_loader, num_epochs, lr, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    best_epoch = 0

    # Prepare log file
    log_path = os.path.join(save_dir, 'log.txt')
    with open(log_path, 'w') as log_file:
        log_file.write(f"Model architecture:\n{model}\n\n")

    # Lists to store metrics
    train_losses, val_losses = [], []
    train_metrics_history = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    val_metrics_history = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    lrs = []

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        running_loss, all_preds, all_labels = 0.0, [], []

        for mouse1, mouse2, mask1, mask2, labels in train_loader:
            mouse1, mouse2, mask1, mask2, labels = [x.to(device) for x in (mouse1, mouse2, mask1, mask2, labels)]
            optimizer.zero_grad()
            outputs = model(mouse1, mouse2, mask1, mask2)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(all_labels)
        train_metrics = calculate_metrics(all_labels, all_preds)

        # Validation
        model.eval()
        val_running_loss, val_all_preds, val_all_labels = 0.0, [], []

        with torch.no_grad():
            for mouse1, mouse2, mask1, mask2, labels in val_loader:
                mouse1, mouse2, mask1, mask2, labels = [x.to(device) for x in (mouse1, mouse2, mask1, mask2, labels)]
                outputs = model(mouse1, mouse2, mask1, mask2)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_all_labels)
        val_metrics = calculate_metrics(val_all_labels, val_all_preds)

        # Scheduler and LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(current_lr)

        for metric in train_metrics.keys():
            train_metrics_history[metric].append(train_metrics[metric] * 100)
            val_metrics_history[metric].append(val_metrics[metric] * 100)

        # Save best model based on accuracy
        if val_metrics['accuracy'] > best_val_metrics['accuracy']:
            best_val_metrics = val_metrics.copy()
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

            # Save confusion matrix for best model
            plot_confusion_matrix(val_all_labels, val_all_preds, save_dir, epoch)

        # Log epoch
        log_line = (f"Epoch {epoch:03d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy'] * 100:.2f}%, "
                    f"P: {train_metrics['precision'] * 100:.2f}%, R: {train_metrics['recall'] * 100:.2f}%, "
                    f"F1: {train_metrics['f1_score'] * 100:.2f}% | "
                    f"Val Acc: {val_metrics['accuracy'] * 100:.2f}%, "
                    f"P: {val_metrics['precision'] * 100:.2f}%, R: {val_metrics['recall'] * 100:.2f}%, "
                    f"F1: {val_metrics['f1_score'] * 100:.2f}% | "
                    f"LR: {current_lr:.6f}\n")
        print(log_line.strip())
        with open(log_path, 'a') as log_file:
            log_file.write(log_line)

    # Plot training curves
    epochs = list(range(1, num_epochs + 1))
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Loss Curve
    axs[0, 0].plot(epochs, train_losses, label='Train Loss')
    axs[0, 0].set_title('Loss Curve')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Accuracy Curve
    axs[0, 1].plot(epochs, train_metrics_history['accuracy'], label='Train Acc')
    axs[0, 1].plot(epochs, val_metrics_history['accuracy'], label='Val Acc')
    axs[0, 1].set_title('Accuracy Curve')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy (%)')
    axs[0, 1].legend()

    # Precision Curve
    axs[0, 2].plot(epochs, train_metrics_history['precision'], label='Train Precision')
    axs[0, 2].plot(epochs, val_metrics_history['precision'], label='Val Precision')
    axs[0, 2].set_title('Precision Curve')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('Precision (%)')
    axs[0, 2].legend()

    # Recall Curve
    axs[1, 0].plot(epochs, train_metrics_history['recall'], label='Train Recall')
    axs[1, 0].plot(epochs, val_metrics_history['recall'], label='Val Recall')
    axs[1, 0].set_title('Recall Curve')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Recall (%)')
    axs[1, 0].legend()

    # F1 Score Curve
    axs[1, 1].plot(epochs, train_metrics_history['f1_score'], label='Train F1')
    axs[1, 1].plot(epochs, val_metrics_history['f1_score'], label='Val F1')
    axs[1, 1].set_title('F1 Score Curve')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('F1 Score (%)')
    axs[1, 1].legend()

    # Learning Rate Curve
    axs[1, 2].plot(epochs, lrs)
    axs[1, 2].set_title('LR Schedule')
    axs[1, 2].set_xlabel('Epoch')
    axs[1, 2].set_ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.pdf'))
    plt.close()

    # Convert best metrics to percentages for return
    for key in best_val_metrics:
        best_val_metrics[key] *= 100

    print(f"\nBest model saved at epoch {best_epoch}")
    print(f"Best validation metrics: Acc: {best_val_metrics['accuracy']:.2f}%, "
          f"P: {best_val_metrics['precision']:.2f}%, R: {best_val_metrics['recall']:.2f}%, "
          f"F1: {best_val_metrics['f1_score']:.2f}%")

    # return best_val_metrics
    return best_val_metrics['accuracy']
