import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


class TwoStreamMLP(nn.Module):
    """MLP baseline for two-stream calcium imaging data"""

    def __init__(self, input_dim=128, seq_length=196, hidden_dims=[512, 256, 128],
                 num_classes=2, dropout=0.5, fusion_strategy='concat'):
        super(TwoStreamMLP, self).__init__()

        self.fusion_strategy = fusion_strategy
        assert fusion_strategy in ['concat', 'add', 'max', 'mean'], \
            "fusion_strategy must be one of: 'concat', 'add', 'max', 'mean'"

        # Feature dimensions after flattening or pooling
        self.feature_dim = input_dim * seq_length if fusion_strategy == 'concat' else input_dim * seq_length

        # Stream 1 MLP
        stream1_layers = []
        prev_dim = self.feature_dim
        for hidden_dim in hidden_dims:
            stream1_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.stream1_mlp = nn.Sequential(*stream1_layers)

        # Stream 2 MLP (same architecture)
        stream2_layers = []
        prev_dim = self.feature_dim
        for hidden_dim in hidden_dims:
            stream2_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.stream2_mlp = nn.Sequential(*stream2_layers)

        # Fusion layer
        if fusion_strategy == 'concat':
            fusion_input_dim = hidden_dims[-1] * 2
        else:
            fusion_input_dim = hidden_dims[-1]

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim // 2, num_classes)
        )

    def forward(self, mouse1_data, mouse2_data, mouse1_mask=None, mouse2_mask=None):
        # Flatten the sequence data
        batch_size = mouse1_data.size(0)

        # Apply masks to zero out padding
        if mouse1_mask is not None:
            mouse1_data = mouse1_data * mouse1_mask.unsqueeze(-1).float()
        if mouse2_mask is not None:
            mouse2_data = mouse2_data * mouse2_mask.unsqueeze(-1).float()

        # Flatten: (batch, seq_len, features) -> (batch, seq_len * features)
        mouse1_flat = mouse1_data.view(batch_size, -1)
        mouse2_flat = mouse2_data.view(batch_size, -1)

        # Pass through respective MLPs
        stream1_features = self.stream1_mlp(mouse1_flat)
        stream2_features = self.stream2_mlp(mouse2_flat)

        # Fusion
        if self.fusion_strategy == 'concat':
            fused_features = torch.cat([stream1_features, stream2_features], dim=1)
        elif self.fusion_strategy == 'add':
            fused_features = stream1_features + stream2_features
        elif self.fusion_strategy == 'max':
            fused_features = torch.max(stream1_features, stream2_features)
        elif self.fusion_strategy == 'mean':
            fused_features = (stream1_features + stream2_features) / 2

        # Final classification
        logits = self.fusion_layer(fused_features)
        return logits


class TwoStreamSVM:
    """SVM baseline for two-stream calcium imaging data"""

    def __init__(self, C=1.0, kernel='rbf', gamma='scale', fusion_strategy='concat'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.fusion_strategy = fusion_strategy
        self.scaler = StandardScaler()
        self.svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)

        assert fusion_strategy in ['concat', 'add', 'max', 'mean'], \
            "fusion_strategy must be one of: 'concat', 'add', 'max', 'mean'"

    def _extract_features(self, dataloader, device='cpu'):
        """Extract features from dataloader for SVM training"""
        all_features = []
        all_labels = []

        with torch.no_grad():
            for mouse1_data, mouse2_data, mouse1_mask, mouse2_mask, labels in tqdm(dataloader,
                                                                                   desc="Extracting features"):
                batch_size = mouse1_data.size(0)

                # Apply masks
                if mouse1_mask is not None:
                    mouse1_data = mouse1_data * mouse1_mask.unsqueeze(-1).float()
                if mouse2_mask is not None:
                    mouse2_data = mouse2_data * mouse2_mask.unsqueeze(-1).float()

                # Flatten features
                mouse1_flat = mouse1_data.view(batch_size, -1).numpy()
                mouse2_flat = mouse2_data.view(batch_size, -1).numpy()

                # Feature fusion
                if self.fusion_strategy == 'concat':
                    fused_features = np.concatenate([mouse1_flat, mouse2_flat], axis=1)
                elif self.fusion_strategy == 'add':
                    fused_features = mouse1_flat + mouse2_flat
                elif self.fusion_strategy == 'max':
                    fused_features = np.maximum(mouse1_flat, mouse2_flat)
                elif self.fusion_strategy == 'mean':
                    fused_features = (mouse1_flat + mouse2_flat) / 2

                all_features.append(fused_features)
                all_labels.append(labels.numpy())

        return np.vstack(all_features), np.concatenate(all_labels)

    def fit(self, train_loader):
        """Train the SVM model"""
        print("Extracting training features...")
        X_train, y_train = self._extract_features(train_loader)

        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        print("Training SVM...")
        self.svm.fit(X_train_scaled, y_train)

        return self

    def predict(self, test_loader):
        """Make predictions on test data"""
        print("Extracting test features...")
        X_test, y_test = self._extract_features(test_loader)

        print("Scaling features...")
        X_test_scaled = self.scaler.transform(X_test)

        print("Making predictions...")
        y_pred = self.svm.predict(X_test_scaled)
        y_pred_proba = self.svm.predict_proba(X_test_scaled)

        return y_pred, y_pred_proba, y_test


    def evaluate(self, test_loader):
        """Evaluate the SVM model"""
        y_pred, y_pred_proba, y_true = self.predict(test_loader)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        report = classification_report(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_true,
            'classification_report': report
        }