import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Set2SetLoss(nn.Module):
    """Set2Set 손실 함수 (Type2Branch 참고)"""
    
    def __init__(self, K=10, N=15, beta=0.05):
        super(Set2SetLoss, self).__init__()
        self.K = K  # 배치당 세트 수
        self.N = N  # 사용자당 샘플 수
        self.beta = beta  # 반지름 페널티 하이퍼파라미터
        
        # 가중치 행렬 p 초기화 (삼각형 패턴)
        p = torch.zeros([N, N, N], dtype=torch.float)
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(N):
                    p[i, j, k] = 1.0
        self.register_buffer('p', p)
    
    def calculate_set2set_loss(self, embeddings):
        """
        Set2Set 손실 계산
        
        Args:
            embeddings: 임베딩 벡터들 [batch_size, embedding_dim]
                       batch_size = K * N
        
        Returns:
            loss: Set2Set 손실
        """
        # 쌍별 거리 계산
        pdist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Set Margin Loss (Lsm) 계산
        set_margin_loss = 0.0
        total_pairs = 0
        
        for i in range(self.K - 1):
            for j in range(i + 1, self.K):
                posA = self.N * i
                posB = self.N * j
                
                # 세트 A 내부 거리
                eA = pdist_matrix[posA:posA + self.N, posA:posA + self.N].unsqueeze(-1)
                eA = eA.repeat(1, 1, self.N)
                
                # 세트 A와 세트 B 간 거리
                eB = pdist_matrix[posA:posA + self.N, posB:posB + self.N].unsqueeze(1)
                eB = eB.repeat(1, self.N, 1)
                
                # 마진 계산
                margin = eA - eB + 1.5
                margin = torch.clamp(margin, min=0)
                
                # 가중치 적용
                margin = margin * self.p
                loss = torch.sum(margin)
                
                set_margin_loss += loss
                total_pairs += 1
        
        # 정규화
        set_margin_loss /= (self.N * self.N * (self.N - 1) / 2)
        set_margin_loss /= total_pairs
        
        # Radius Penalty (Lrp) 계산
        total_radius = 0.0
        for i in range(self.K):
            legitimate_embeddings = embeddings[self.N * i:self.N * i + self.N]
            centroid = torch.mean(legitimate_embeddings, dim=0)
            distances = torch.norm(legitimate_embeddings - centroid, dim=1)
            mean_distance = torch.mean(distances)
            total_radius += mean_distance
        
        total_radius /= self.K
        
        # 반지름 페널티 계산
        radius_penalty = 0.0
        for i in range(self.K):
            legitimate_embeddings = embeddings[self.N * i:self.N * i + self.N]
            centroid = torch.mean(legitimate_embeddings, dim=0)
            distances = torch.norm(legitimate_embeddings - centroid, dim=1)
            mean_distance = torch.mean(distances)
            radius_penalty += torch.abs(mean_distance / total_radius - 1.0)
        
        radius_penalty /= self.K
        radius_penalty *= self.beta
        
        # 최종 손실
        total_loss = set_margin_loss + radius_penalty
        
        return total_loss
    
    def forward(self, embeddings):
        """
        순전파
        
        Args:
            embeddings: 임베딩 벡터들 [batch_size, embedding_dim]
        
        Returns:
            loss: Set2Set 손실
        """
        return self.calculate_set2set_loss(embeddings)

class SimplifiedSet2SetLoss(nn.Module):
    """단순화된 Set2Set 손실 함수"""
    
    def __init__(self, margin=1.5, beta=0.05):
        super(SimplifiedSet2SetLoss, self).__init__()
        self.margin = margin
        self.beta = beta
    
    def forward(self, anchor, positive, negative):
        """
        단순화된 Set2Set 손실 계산
        
        Args:
            anchor: 앵커 샘플 [batch_size, embedding_dim]
            positive: 포지티브 샘플 [batch_size, embedding_dim]
            negative: 네거티브 샘플 [batch_size, embedding_dim]
        
        Returns:
            loss: 단순화된 Set2Set 손실
        """
        # 앵커와 포지티브 간 거리
        distance_positive = torch.norm(anchor - positive, dim=1)
        
        # 앵커와 네거티브 간 거리
        distance_negative = torch.norm(anchor - negative, dim=1)
        
        # Set Margin Loss
        set_margin_loss = F.relu(distance_positive - distance_negative + self.margin)
        
        # Radius Penalty (각 세트의 중심점과의 거리 일관성)
        anchor_centroid = (anchor + positive) / 2
        anchor_to_centroid = torch.norm(anchor - anchor_centroid, dim=1)
        positive_to_centroid = torch.norm(positive - anchor_centroid, dim=1)
        
        # 반지름 일관성 페널티
        radius_penalty = torch.abs(anchor_to_centroid - positive_to_centroid)
        
        # 최종 손실
        total_loss = set_margin_loss.mean() + self.beta * radius_penalty.mean()
        
        return total_loss

class ContrastiveSet2SetLoss(nn.Module):
    """대조 학습 기반 Set2Set 손실 함수"""
    
    def __init__(self, temperature=0.1, margin=1.0):
        super(ContrastiveSet2SetLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        대조 학습 기반 Set2Set 손실 계산
        
        Args:
            anchor: 앵커 샘플 [batch_size, embedding_dim]
            positive: 포지티브 샘플 [batch_size, embedding_dim]
            negative: 네거티브 샘플 [batch_size, embedding_dim]
        
        Returns:
            loss: 대조 학습 기반 Set2Set 손실
        """
        # 정규화
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        
        # 유사도 계산
        sim_positive = torch.sum(anchor * positive, dim=1) / self.temperature
        sim_negative = torch.sum(anchor * negative, dim=1) / self.temperature
        
        # 대조 손실
        contrastive_loss = -torch.log(
            torch.exp(sim_positive) / (torch.exp(sim_positive) + torch.exp(sim_negative))
        )
        
        # 마진 기반 손실
        margin_loss = F.relu(sim_negative - sim_positive + self.margin)
        
        # 최종 손실
        total_loss = contrastive_loss.mean() + margin_loss.mean()
        
        return total_loss

def get_set2set_loss(loss_type='standard', **kwargs):
    """Set2Set 손실 함수 생성"""
    
    if loss_type == 'standard':
        return Set2SetLoss(**kwargs)
    elif loss_type == 'simplified':
        return SimplifiedSet2SetLoss(**kwargs)
    elif loss_type == 'contrastive':
        return ContrastiveSet2SetLoss(**kwargs)
    else:
        raise ValueError(f"Unknown Set2Set loss type: {loss_type}") 