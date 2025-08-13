import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """트리플렛 손실 함수"""
    
    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def calc_distance(self, x1, x2):
        """거리 계산"""
        if self.distance_metric == 'euclidean':
            return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))
        elif self.distance_metric == 'cosine':
            return 1 - F.cosine_similarity(x1, x2, dim=1)
        elif self.distance_metric == 'manhattan':
            return torch.sum(torch.abs(x1 - x2), dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(self, anchor, positive, negative):
        """
        트리플렛 손실 계산
        
        Args:
            anchor: 앵커 샘플 [batch_size, embedding_dim]
            positive: 포지티브 샘플 [batch_size, embedding_dim]
            negative: 네거티브 샘플 [batch_size, embedding_dim]
            
        Returns:
            loss: 트리플렛 손실
        """
        # 거리 계산
        distance_positive = self.calc_distance(anchor, positive)
        distance_negative = self.calc_distance(anchor, negative)
        
        # 트리플렛 손실 계산
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean()

class HardTripletLoss(nn.Module):
    """하드 트리플렛 손실 함수 (가장 어려운 네거티브 선택)"""
    
    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def calc_distance(self, x1, x2):
        """거리 계산"""
        if self.distance_metric == 'euclidean':
            return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))
        elif self.distance_metric == 'cosine':
            return 1 - F.cosine_similarity(x1, x2, dim=1)
        elif self.distance_metric == 'manhattan':
            return torch.sum(torch.abs(x1 - x2), dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(self, anchor, positive, negatives):
        """
        하드 트리플렛 손실 계산
        
        Args:
            anchor: 앵커 샘플 [batch_size, embedding_dim]
            positive: 포지티브 샘플 [batch_size, embedding_dim]
            negatives: 네거티브 샘플들 [batch_size, num_negatives, embedding_dim]
            
        Returns:
            loss: 하드 트리플렛 손실
        """
        batch_size = anchor.size(0)
        
        # 앵커와 포지티브 간 거리
        distance_positive = self.calc_distance(anchor, positive)
        
        # 앵커와 모든 네거티브 간 거리
        anchor_expanded = anchor.unsqueeze(1).expand(-1, negatives.size(1), -1)
        distance_negatives = self.calc_distance(
            anchor_expanded.reshape(-1, anchor.size(1)),
            negatives.reshape(-1, negatives.size(-1))
        ).reshape(batch_size, -1)
        
        # 가장 어려운 네거티브 선택 (가장 가까운 네거티브)
        hardest_negative_distance = torch.min(distance_negatives, dim=1)[0]
        
        # 하드 트리플렛 손실 계산
        losses = F.relu(distance_positive - hardest_negative_distance + self.margin)
        
        return losses.mean()

class SemiHardTripletLoss(nn.Module):
    """세미-하드 트리플렛 손실 함수"""
    
    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super(SemiHardTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def calc_distance(self, x1, x2):
        """거리 계산"""
        if self.distance_metric == 'euclidean':
            return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))
        elif self.distance_metric == 'cosine':
            return 1 - F.cosine_similarity(x1, x2, dim=1)
        elif self.distance_metric == 'manhattan':
            return torch.sum(torch.abs(x1 - x2), dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(self, anchor, positive, negatives):
        """
        세미-하드 트리플렛 손실 계산
        
        Args:
            anchor: 앵커 샘플 [batch_size, embedding_dim]
            positive: 포지티브 샘플 [batch_size, embedding_dim]
            negatives: 네거티브 샘플들 [batch_size, num_negatives, embedding_dim]
            
        Returns:
            loss: 세미-하드 트리플렛 손실
        """
        batch_size = anchor.size(0)
        
        # 앵커와 포지티브 간 거리
        distance_positive = self.calc_distance(anchor, positive)
        
        # 앵커와 모든 네거티브 간 거리
        anchor_expanded = anchor.unsqueeze(1).expand(-1, negatives.size(1), -1)
        distance_negatives = self.calc_distance(
            anchor_expanded.reshape(-1, anchor.size(1)),
            negatives.reshape(-1, negatives.size(-1))
        ).reshape(batch_size, -1)
        
        # 세미-하드 네거티브 선택
        # distance_positive < distance_negative < distance_positive + margin 조건을 만족하는 네거티브
        semi_hard_mask = (distance_positive.unsqueeze(1) < distance_negatives) & \
                        (distance_negatives < distance_positive.unsqueeze(1) + self.margin)
        
        # 세미-하드 네거티브가 있는 경우에만 손실 계산
        valid_triplets = semi_hard_mask.any(dim=1)
        
        if valid_triplets.sum() == 0:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)
        
        # 세미-하드 네거티브 중 가장 어려운 것 선택
        semi_hard_distances = distance_negatives.clone()
        semi_hard_distances[~semi_hard_mask] = float('inf')
        hardest_semi_hard_distance = torch.min(semi_hard_distances, dim=1)[0]
        
        # 손실 계산
        losses = F.relu(distance_positive - hardest_semi_hard_distance + self.margin)
        losses = losses[valid_triplets]
        
        return losses.mean()

def get_triplet_loss(loss_type='standard', margin=1.0, distance_metric='euclidean'):
    """트리플렛 손실 함수 생성"""
    
    if loss_type == 'standard':
        return TripletLoss(margin=margin, distance_metric=distance_metric)
    elif loss_type == 'hard':
        return HardTripletLoss(margin=margin, distance_metric=distance_metric)
    elif loss_type == 'semi_hard':
        return SemiHardTripletLoss(margin=margin, distance_metric=distance_metric)
    else:
        raise ValueError(f"Unknown triplet loss type: {loss_type}") 