import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """학습 가능한 가우시안 혼합 모델 기반 위치 인코딩"""
    
    def __init__(self, k, d_model, seq_len):
        super().__init__()
        
        # 학습 가능한 임베딩
        self.embedding = nn.Parameter(torch.zeros([k, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        
        # 위치 정보
        self.register_buffer('positions', torch.tensor([i for i in range(seq_len)], dtype=torch.float).unsqueeze(1).repeat(1, k))
        
        # 가우시안 혼합 모델의 파라미터들 (형상 안정화)
        # mu: (1, k) - seq_len 범위에서 균등 간격 초기화
        self.mu = nn.Parameter(torch.linspace(0, seq_len - 1, steps=k, dtype=torch.float).unsqueeze(0), requires_grad=True)
        # sigma_raw: (1, k) - softplus로 양수 보장 (초기값 ≈ 50)
        self.sigma_raw = nn.Parameter(torch.full((1, k), 3.9120))  # softplus(3.9120) ≈ 50
        
    def normal_pdf(self, pos, mu, sigma):
        """정규 분포 PDF 계산"""
        a = pos - mu
        log_p = -1 * torch.mul(a, a) / (2 * (sigma ** 2)) - torch.log(sigma)
        return torch.nn.functional.softmax(log_p, dim=1)

    def forward(self, inputs):
        """순전파"""
        # sigma를 안정적으로 양수로 변환
        sigma = F.softplus(self.sigma_raw) + 1e-6  # (1, k)
        pdfs = self.normal_pdf(self.positions, self.mu, sigma)  # positions: (seq_len, k)
        pos_enc = torch.matmul(pdfs, self.embedding)
        
        return inputs + pos_enc.unsqueeze(0).repeat(inputs.size(0), 1, 1)

class TemporalAttention(nn.Module):
    """시간적 어텐션 메커니즘"""
    
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Q, K, V 계산
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 어텐션 스코어 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 어텐션 적용
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.output(attended)

class ChannelAttention(nn.Module):
    """채널 어텐션 메커니즘"""
    
    def __init__(self, d_model, reduction_ratio=16):
        super().__init__()
        self.d_model = d_model
        self.reduction_ratio = reduction_ratio
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // reduction_ratio),
            nn.ReLU(),
            nn.Linear(d_model // reduction_ratio, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # 평균 및 최대 풀링
        avg_out = self.avg_pool(x.transpose(1, 2)).squeeze(-1)
        max_out = self.max_pool(x.transpose(1, 2)).squeeze(-1)
        
        # 채널 어텐션 가중치 계산
        attention = self.fc(avg_out) + self.fc(max_out)
        attention = attention.unsqueeze(1)
        
        # 어텐션 적용
        return x * attention

class TransformerEncoderLayer(nn.Module):
    """트랜스포머 인코더 레이어"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DualAttentionBlock(nn.Module):
    """이중 어텐션 블록 (Temporal + Channel)"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.temporal_attention = TemporalAttention(d_model, num_heads)
        self.channel_attention = ChannelAttention(d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Temporal Attention
        temp_output = self.temporal_attention(x)
        x = self.norm1(x + self.dropout(temp_output))
        
        # Channel Attention
        chan_output = self.channel_attention(x)
        x = self.norm2(x + self.dropout(chan_output))
        
        return x

class KeystrokeTransformer(nn.Module):
    """키스트로크 인증을 위한 Transformer + BiGRU 하이브리드 모델"""
    
    def __init__(self, input_dim=3, seq_len=50, d_model=256, num_layers=6, 
                 num_heads=8, hidden_size=128, target_len=64, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.target_len = target_len
        
        # 입력 프로젝션
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(20, d_model, seq_len)
        
        # 트랜스포머 인코더 레이어들
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # BiGRU 레이어
        self.bigru = nn.GRU(d_model, hidden_size, bidirectional=True, batch_first=True)
        
        # 이중 어텐션 블록
        self.dual_attention = DualAttentionBlock(d_model, num_heads, dropout)
        
        # 출력 프로젝션
        self.output_projection = nn.Sequential(
            nn.Linear(d_model + hidden_size * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, target_len),
            nn.ReLU()
        )
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 입력 형태: [batch_size, seq_len, input_dim]
        batch_size = x.size(0)
        
        # 입력 프로젝션
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 위치 인코딩 적용
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 트랜스포머 인코더 레이어들
        for layer in self.transformer_layers:
            x = layer(x)
        
        # BiGRU 처리
        gru_out, _ = self.bigru(x)  # [batch_size, seq_len, hidden_size * 2]
        
        # 이중 어텐션 적용
        attended = self.dual_attention(x)  # [batch_size, seq_len, d_model]
        
        # 특징 결합
        combined = torch.cat([attended, gru_out], dim=-1)  # [batch_size, seq_len, d_model + hidden_size * 2]
        
        # 글로벌 평균 풀링
        pooled = torch.mean(combined, dim=1)  # [batch_size, d_model + hidden_size * 2]
        
        # 출력 프로젝션
        output = self.output_projection(pooled)  # [batch_size, target_len]
        # 임베딩 L2 정규화로 수치 안정성 강화
        output = F.normalize(output, dim=-1)
        
        return output

def create_model(config):
    """설정에 따라 모델 생성"""
    hyperparams = config.get_config_dict()["hyperparams"]
    
    model = KeystrokeTransformer(
        input_dim=hyperparams["keystroke_feature_count"],
        seq_len=config.get_config_dict()["data"]["keystroke_sequence_len"],
        d_model=hyperparams["d_model"],
        num_layers=hyperparams["num_layers"],
        num_heads=hyperparams["num_heads"],
        hidden_size=hyperparams["hidden_size"],
        target_len=hyperparams["target_len"],
        dropout=hyperparams["dropout"]
    )
    
    return model

if __name__ == "__main__":
    # 모델 테스트
    from utils.config import Config
    
    config = Config()
    model = create_model(config)
    
    # 테스트 입력
    batch_size = 4
    seq_len = 50
    input_dim = 3
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}") 