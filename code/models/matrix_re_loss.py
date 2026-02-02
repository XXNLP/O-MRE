import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np


class MatrixRELoss(nn.Module):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()

    def forward(self, score, predicate_one_hot_labels):
        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float()

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float()  # BS, NL

        loss = ((F.binary_cross_entropy(score, predicate_one_hot_labels, reduction="none") * entity_mask).sum(dim=(2, 3)) / entity_sum).mean()
        if loss.item() < 0:
            print("debug")
        return loss


class BottleneckMatrixRELoss(nn.Module):
    """
    Bottleneck Information Loss for sentence-level relation extraction.
    """

    def __init__(self, beta=1e-3):
        """
        Args:
            beta: Weight for the compression term (I(Z; X)).
        """
        super().__init__()
        self.beta = beta  # Weight for the compression term

    def forward(self, score, predicate_one_hot_labels, latent_image_features, latent_text_features):
        """
        Forward pass for the Bottleneck Information Loss.

        Args:
            score: Predicted scores (e.g., logits after sigmoid) [BS, NL, H, W].
            predicate_one_hot_labels: Ground-truth labels [BS, NL, H, W].
            latent_representation: Latent representation Z [BS, D].
            prior_distribution: Prior distribution for Z (e.g., standard normal) [BS, D].

        Returns:
            loss: Computed loss value.
        """
        # Prediction Loss (maximize I(Z; Y))
        """
        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float()

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float()  # BS, NL
        """

        # Compute entity mask
        device = score.device
        predicate_one_hot_labels = predicate_one_hot_labels.to(device)

        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float().to(device)

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float().to(device)  # BS, NL

        # Prediction loss (binary cross-entropy)
        prediction_loss = ((F.binary_cross_entropy(score, predicate_one_hot_labels, reduction="none") * entity_mask).sum(dim=(2, 3)) / entity_sum).mean()


        attention_bottleneck = AttentionBottleneck(feature_dim=latent_image_features.size(-1))
        weighted_image_features, image_weights = attention_bottleneck(latent_image_features)
        weighted_text_features, text_weights = attention_bottleneck(latent_text_features)

        # 注意力熵正则化
        entropy_loss = -torch.mean(image_weights * torch.log(image_weights + 1e-6)) \
                    - torch.mean(text_weights * torch.log(text_weights + 1e-6))

        # 总损失
        total_loss = prediction_loss + self.beta * entropy_loss

        return total_loss 


import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixRELossWithMiddleLayer(nn.Module):
    """
    Bottleneck Information Loss for multi-modal relation extraction using middle-layer features from CLIP.
    """

    def __init__(self, beta=1e-3):
        """
        Args:
            beta: Weight for the bottleneck regularization term.
        """
        super().__init__()
        self.beta = beta  # Weight for bottleneck term

    def forward(self, score, predicate_one_hot_labels, latent_image_features, latent_text_features):
        """
        Args:
            score: Predicted score (BS, NL, ..., C).
            predicate_one_hot_labels: Ground truth one-hot labels (BS, NL, ...).
            latent_image_features: Middle-layer features from CLIP's image encoder.
            latent_text_features: Middle-layer features from CLIP's text encoder.

        Returns:
            loss: Combined loss with bottleneck information.
        """
        # Compute entity mask
        #device = score.device
        #predicate_one_hot_labels = predicate_one_hot_labels.to(device)

        #entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        #entity_mask = (entity_mask > 0).float().to(device)

        #entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float().to(device)  # BS, NL


        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float()

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float()  # BS, NL

        # Prediction loss (binary cross-entropy)
        prediction_loss = ((F.binary_cross_entropy(score, predicate_one_hot_labels, reduction="none") * entity_mask).sum(dim=(2, 3)) / entity_sum).mean()

        # Bottleneck regularization term
        # Compress image and text latent representations to approximate mutual information regularization
        compression_loss_image = torch.mean(torch.norm(latent_image_features, dim=-1))  # L2 norm for image features
        compression_loss_text = torch.mean(torch.norm(latent_text_features, dim=-1))    # L2 norm for text features

        # Total compression loss
        compression_loss = compression_loss_image + compression_loss_text

        # Total loss
        loss = prediction_loss + self.beta * compression_loss

        return loss

# 鼓励不同模态的特征（图像和文本）在语义相关时靠近，在语义无关时分开
# Contrastive Regularization
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        labels = torch.arange(len(image_features)).to(image_features.device)
        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss

class AttentionBottleneck(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        attention_weights = self.attention(features)
        return features * attention_weights, attention_weights


class BottleneckLoss(nn.Module):
    """
    Bottleneck Information Loss for multi-modal relation extraction using middle-layer features from CLIP.
    """

    def __init__(self, beta=1e-3,BIAttentitonDim=768):
        """
        Args:
            beta: Weight for the bottleneck regularization term.
        """
        super().__init__()
        self.beta = beta  # Weight for bottleneck term
        self.attention_bottleneck = AttentionBottleneck(feature_dim=BIAttentitonDim)

    def forward(self, latent_image_features, latent_text_features):
        """
        Args:
            score: Predicted score (BS, NL, ..., C).
            predicate_one_hot_labels: Ground truth one-hot labels (BS, NL, ...).
            latent_image_features: Middle-layer features from CLIP's image encoder.
            latent_text_features: Middle-layer features from CLIP's text encoder.

        Returns:
            loss: Combined loss with bottleneck information.
        """
        weighted_image_features, image_weights = self.attention_bottleneck(latent_image_features)
        weighted_text_features, text_weights = self.attention_bottleneck(latent_text_features)

        # 注意力熵正则化
        entropy_loss = -torch.mean(image_weights * torch.log(image_weights + 1e-6)) \
                    - torch.mean(text_weights * torch.log(text_weights + 1e-6))
        #目标是让绝大多数 image_weights or text_weights 接近 0，而少数α较大。
        # 总损失
        total_loss = self.beta * entropy_loss

        return total_loss 
    


class BottleneckLossNorm(nn.Module):
    """
    Bottleneck Information Loss for multi-modal relation extraction using middle-layer features from CLIP.
    """

    def __init__(self, beta=1e-3):
        """
        Args:
            beta: Weight for the bottleneck regularization term.
        """
        super().__init__()
        self.beta = beta  # Weight for bottleneck term

    def forward(self, latent_image_features, latent_text_features):
        """
        Args:
            score: Predicted score (BS, NL, ..., C).
            predicate_one_hot_labels: Ground truth one-hot labels (BS, NL, ...).
            latent_image_features: Middle-layer features from CLIP's image encoder.
            latent_text_features: Middle-layer features from CLIP's text encoder.

        Returns:
            loss: Combined loss with bottleneck information.
        """

        # Bottleneck regularization term
        # Compress image and text latent representations to approximate mutual information regularization
        compression_loss_image = torch.mean(torch.norm(latent_image_features, dim=-1))  # L2 norm for image features
        compression_loss_text = torch.mean(torch.norm(latent_text_features, dim=-1))    # L2 norm for text features

        # Total compression loss
        compression_loss = compression_loss_image + compression_loss_text

        # Total loss
        loss = self.beta * compression_loss

        return loss
    
# 最好的结果发生在这里，其中attention是随机生成的
class MatrixRELossWithMiddleLayerAttentionBI(nn.Module):
    """
    Bottleneck Information Loss for sentence-level relation extraction.
    """

    def __init__(self, beta=1e-3,BIAttentitonDim=768):
        """
        Args:
            beta: Weight for the compression term (I(Z; X)).
        """
        super().__init__()
        self.beta = beta  # Weight for the compression term
        self.attention_bottleneck = AttentionBottleneck(feature_dim=BIAttentitonDim)
        
    def forward(self, score, predicate_one_hot_labels, latent_image_features, latent_text_features):
        """
        Forward pass for the Bottleneck Information Loss.

        Args:
            score: Predicted scores (e.g., logits after sigmoid) [BS, NL, H, W].
            predicate_one_hot_labels: Ground-truth labels [BS, NL, H, W].
            latent_representation: Latent representation Z [BS, D].
            prior_distribution: Prior distribution for Z (e.g., standard normal) [BS, D].

        Returns:
            loss: Computed loss value.
        """
        # Prediction Loss (maximize I(Z; Y))
        """
        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float()

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float()  # BS, NL
        """

        # Compute entity mask
        device = score.device
        predicate_one_hot_labels = predicate_one_hot_labels.to(device)

        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float().to(device)

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float().to(device)  # BS, NL
        
        # Prediction loss (binary cross-entropy)
        prediction_loss = ((F.binary_cross_entropy(score, predicate_one_hot_labels, reduction="none") * entity_mask).sum(dim=(2, 3)) / entity_sum).mean()


        #attention_bottleneck = AttentionBottleneck(latent_image_features.shape[-1])
        weighted_image_features, image_weights = self.attention_bottleneck(latent_image_features)
        weighted_text_features, text_weights = self.attention_bottleneck(latent_text_features)

        # 注意力熵正则化
        entropy_loss = -torch.mean(image_weights * torch.log(image_weights + 1e-6)) \
                    - torch.mean(text_weights * torch.log(text_weights + 1e-6))

        # 总损失
        total_loss = prediction_loss + self.beta * entropy_loss

        return total_loss 

# 尝试用两个attention loss 分别处理不同特征
class MatrixRELossWithMiddleLayerAttentionBI_atts(nn.Module):
    """
    Bottleneck Information Loss for sentence-level relation extraction.
    """

    def __init__(self, beta=1e-3,BIAttentitonDim=768):
        """
        Args:
            beta: Weight for the compression term (I(Z; X)).
        """
        super().__init__()
        self.beta = beta  # Weight for the compression term
        self.text_attention_bottleneck = AttentionBottleneck(feature_dim=BIAttentitonDim)
        self.visio_attention_bottleneck = AttentionBottleneck(feature_dim=BIAttentitonDim)
        
    def forward(self, score, predicate_one_hot_labels, latent_image_features, latent_text_features):
        """
        Forward pass for the Bottleneck Information Loss.

        Args:
            score: Predicted scores (e.g., logits after sigmoid) [BS, NL, H, W].
            predicate_one_hot_labels: Ground-truth labels [BS, NL, H, W].
            latent_representation: Latent representation Z [BS, D].
            prior_distribution: Prior distribution for Z (e.g., standard normal) [BS, D].

        Returns:
            loss: Computed loss value.
        """
        # Prediction Loss (maximize I(Z; Y))
        """
        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float()

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float()  # BS, NL
        """

        # Compute entity mask
        device = score.device
        predicate_one_hot_labels = predicate_one_hot_labels.to(device)

        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float().to(device)

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float().to(device)  # BS, NL
        
        # Prediction loss (binary cross-entropy)
        prediction_loss = ((F.binary_cross_entropy(score, predicate_one_hot_labels, reduction="none") * entity_mask).sum(dim=(2, 3)) / entity_sum).mean()


        #attention_bottleneck = AttentionBottleneck(latent_image_features.shape[-1])
        weighted_image_features, image_weights = self.visio_attention_bottleneck(latent_image_features)
        weighted_text_features, text_weights = self.text_attention_bottleneck(latent_text_features)

        # 注意力熵正则化
        entropy_loss = -torch.mean(image_weights * torch.log(image_weights + 1e-6)) \
                    - torch.mean(text_weights * torch.log(text_weights + 1e-6))

        # 总损失
        total_loss = prediction_loss + self.beta * entropy_loss

        return total_loss 
    
#vibloss
class MatrixRELossWithMiddleLayerAttentionBI_vib(nn.Module):
    """
    Bottleneck Information Loss for sentence-level relation extraction.
    """

    def __init__(self, beta=1e-3,BIAttentitonDim=768):
        """
        Args:
            beta: Weight for the compression term (I(Z; X)).
        """
        super().__init__()
        self.beta = beta  # Weight for the compression term
        
        self.vibloss = VIBLoss()
        
    def forward(self, score, predicate_one_hot_labels, latent_image_features, latent_text_features):
        """
        Forward pass for the Bottleneck Information Loss.

        Args:
            score: Predicted scores (e.g., logits after sigmoid) [BS, NL, H, W].
            predicate_one_hot_labels: Ground-truth labels [BS, NL, H, W].
            latent_representation: Latent representation Z [BS, D].
            prior_distribution: Prior distribution for Z (e.g., standard normal) [BS, D].

        Returns:
            loss: Computed loss value.
        """
        # Prediction Loss (maximize I(Z; Y))
        """
        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float()

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float()  # BS, NL
        """

        # Compute entity mask
        device = score.device
        predicate_one_hot_labels = predicate_one_hot_labels.to(device)

        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float().to(device)

        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float().to(device)  # BS, NL
        
        # Prediction loss (binary cross-entropy)
        prediction_loss = ((F.binary_cross_entropy(score, predicate_one_hot_labels, reduction="none") * entity_mask).sum(dim=(2, 3)) / entity_sum).mean()


        # 参数化 q(Z|X)
        text_mu = latent_text_features  # 中间层特征均值
        visio_mu = latent_image_features
        text_logvar = torch.zeros_like(text_mu)  # 可训练变量，用于控制方差
        visio_logvar = torch.zeros_like(visio_mu)  # 可训练变量，用于控制方差

        t_z = text_mu + torch.exp(0.5 * text_logvar) * torch.randn_like(text_mu)  # 重参数化采样
        v_z = visio_mu + torch.exp(0.5 * visio_logvar) * torch.randn_like(visio_mu)  # 重参数化采样
        
        v_vibloss = self.vibloss(latent_image_features, visio_logvar)
        t_vibloss = self.vibloss(latent_text_features, text_logvar)


        # 总损失
        total_loss = prediction_loss + v_vibloss + t_vibloss

        return total_loss 

class VIBLoss(nn.Module):
    def __init__(self, beta=1e-4):
        super().__init__()
        self.beta = beta

    def forward(self, mu, logvar):
        # KL divergence between q(Z|X) and N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        # Combined loss
        return self.beta * kl_loss

