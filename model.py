"""
model.py - Deep Learning Video Violence Detection Architecture
Spatial-Temporal Neural Network (CNN + BiLSTM/GRU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Depthwise separable convolution block for efficient feature extraction."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class StandaloneCNNBackbone(nn.Module):
    """
    Lightweight, high-speed CNN backbone implemented in pure PyTorch.
    Fast on CPU and Laptop GPUs (e.g., RTX 3050).
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            # Initial standard convolution
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            # Efficient depthwise separable layers
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128, stride=1),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, feature_dim, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)


class ViolenceDetectionModel(nn.Module):
    """
    Spatial-Temporal Video Violence Detection Model:
    1. Spatial Feature Extractor (CNN backbone per frame)
    2. Temporal Sequence Model (Bidirectional LSTM or GRU across frames)
    3. Attention-based temporal pooling
    4. Classification Head with Dropout
    """
    def __init__(
        self,
        feature_dim=256,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        use_torchvision=True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Backbone: MobileNetV2 if torchvision is available, otherwise StandaloneCNNBackbone
        self.backbone = None
        if use_torchvision:
            try:
                import torchvision.models as models
                mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
                # Replace classifier with identity to extract 1280-dim feature vector
                mobilenet.classifier = nn.Identity()
                self.backbone = mobilenet
                self.feature_dim = 1280
            except Exception:
                self.backbone = None

        if self.backbone is None:
            self.backbone = StandaloneCNNBackbone(feature_dim=feature_dim)
            self.feature_dim = feature_dim

        # Temporal Sequence Processing (BiLSTM)
        self.temporal_model = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention layer over temporal states
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Binary logit
        )

    def extract_features(self, x):
        """
        Extract spatial features for a batch of frames.
        Args:
            x: (N, C, H, W)
        Returns:
            features: (N, feature_dim)
        """
        return self.backbone(x)

    def classify_features(self, features):
        """
        Sequence classification from extracted features.
        Args:
            features: (B, T, feature_dim)
        Returns:
            logits: (B, 1)
        """
        lstm_out, _ = self.temporal_model(features)
        att_weights = self.attention(lstm_out)
        att_weights = F.softmax(att_weights, dim=1)
        context = torch.sum(lstm_out * att_weights, dim=1)
        return self.classifier(context)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C, H, W)
               B = batch size, T = num_frames (e.g. 16), C = 3, H = 112/224, W = 112/224
        Returns:
            logits: Tensor of shape (B, 1)
        """
        B, T, C, H, W = x.shape

        # Fold time into batch dimension for fast CNN feature extraction
        x = x.contiguous().view(B * T, C, H, W)
        features = self.extract_features(x)

        # Reshape back to sequence
        features = features.view(B, T, self.feature_dim)

        return self.classify_features(features)

    def predict_proba(self, x):
        """Returns probability of violence in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs.squeeze(-1)
