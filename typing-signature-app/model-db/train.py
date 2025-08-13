import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import pickle
import time
import os
import sys
from pathlib import Path

# 경로 설정
sys.path.append(str((Path(__file__)/"../utils").resolve()))

from config import Config
from model import create_model
from data_augmentation import CurriculumGenerator, TestDataset
from losses.triplet_loss import get_triplet_loss
from losses.set2set_loss import get_set2set_loss
import math

class TrainingCallback:
    """훈련 콜백 클래스"""
    
    def __init__(self, model, optimizer, save_path, patience=40):
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.patience = patience
        self.best_eer = float('inf')
        self.counter = 0
        
    def on_epoch_end(self, epoch, loss, eer):
        """에포크 종료 시 호출"""
        is_finite_eer = isinstance(eer, (float, int)) and math.isfinite(float(eer))
        eer_str = f"{eer:.4f}" if is_finite_eer else "skipped"
        print(f"Epoch {epoch}: Loss = {loss:.6f}, EER = {eer_str}")
        
        # 모델 저장 (유효한 EER일 때만)
        if is_finite_eer and eer < self.best_eer:
            self.best_eer = eer
            self.counter = 0
            torch.save(self.model, f"{self.save_path}/best_model_eer_{eer:.4f}.pt")
            print(f"Model saved! New best EER: {eer:.4f}")
        else:
            self.counter += 1
            
        # 체크포인트 저장 (50 에포크마다)
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'eer': eer,
                'best_eer': self.best_eer
            }, f"{self.save_path}/checkpoint_epoch_{epoch+1}.tar")
            
        # 조기 종료 체크
        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs without improvement")
            return True
            
        return False

def load_data():
    """데이터 로딩"""
    print("Loading preprocessed data...")
    
    # 훈련 데이터 로딩
    with open("data/training_data.pickle", 'rb') as f:
        training_data = pickle.load(f)
    
    # 검증 데이터 로딩
    with open("data/validation_data.pickle", 'rb') as f:
        validation_data = pickle.load(f)
    
    print(f"Training users: {len(training_data)}")
    print(f"Validation users: {len(validation_data)}")
    
    return training_data, validation_data

def evaluate_model(model, validation_data, config, device):
    """모델 평가"""
    model.eval()
    
    with torch.no_grad():
        # 검증 데이터 준비
        val_dataset = TestDataset(validation_data, seq_len=config.get_config_dict()["data"]["keystroke_sequence_len"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get_config_dict()["hyperparams"]["batch_size"]["aalto"],
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # 임베딩 추출
        embeddings = []
        for batch in val_loader:
            batch = batch.float().to(device, non_blocking=True)
            with autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
                batch_embeddings = model(batch)
            embeddings.append(batch_embeddings)
        
        embeddings = torch.cat(embeddings, dim=0)
        
        # EER 계산 (간단한 버전)
        num_users = len(validation_data)
        num_sessions = len(validation_data[0])
        
        # 사용자별 임베딩 재구성
        user_embeddings = embeddings.view(num_users, num_sessions, -1)
        
        # 간단한 EER 계산 (실제로는 더 복잡한 계산 필요)
        eer = calculate_simple_eer(user_embeddings)
        
    return eer

def calculate_simple_eer(user_embeddings):
    """간단한 EER 계산"""
    num_users = user_embeddings.size(0)
    num_sessions = user_embeddings.size(1)
    
    # 등록 세션과 검증 세션 분리 (config에서 가져온 값 사용)
    enroll_sessions = 10  # 원래 설정값으로 복원
    verify_sessions = 5   # 원래 설정값으로 복원
    
    # 세션 수 검증
    if enroll_sessions + verify_sessions > num_sessions:
        print(f"Warning: Not enough sessions. Available: {num_sessions}, Required: {enroll_sessions + verify_sessions}")
        return float('inf')  # 세션 수가 부족하면 inf 반환
    
    total_scores = []
    total_labels = []
    
    for user_idx in range(num_users):
        # 등록 임베딩 (평균)
        enroll_emb = torch.mean(user_embeddings[user_idx, :enroll_sessions], dim=0)
        
        # 합법 사용자 검증
        for session_idx in range(enroll_sessions, enroll_sessions + verify_sessions):
            verify_emb = user_embeddings[user_idx, session_idx]
            diff = enroll_emb - verify_emb
            distance = torch.sum(diff * diff).clamp_min(1e-8).sqrt()  # 안정화
            total_scores.append(distance.item())
            total_labels.append(1)  # 합법 사용자
        
        # 불법 사용자 검증 (샘플링으로 최적화)
        other_users = [i for i in range(num_users) if i != user_idx]
        # 최대 5명의 불법 사용자만 샘플링 (안정화 및 속도)
        sampled_impostors = np.random.choice(other_users, size=min(5, len(other_users)), replace=False)
        
        for other_user in sampled_impostors:
            for session_idx in range(enroll_sessions, enroll_sessions + verify_sessions):
                verify_emb = user_embeddings[other_user, session_idx]
                diff = enroll_emb - verify_emb
                distance = torch.sum(diff * diff).clamp_min(1e-8).sqrt()
                total_scores.append(distance.item())
                total_labels.append(0)  # 불법 사용자
    
    # EER 계산 (간단한 버전)
    scores = np.array(total_scores)
    labels = np.array(total_labels)
    
    # 데이터 검증
    if len(scores) == 0 or np.sum(labels == 0) == 0 or np.sum(labels == 1) == 0:
        print("Warning: Insufficient data for EER calculation")
        return float('inf')
    
    # 임계값별 FAR, FRR 계산
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    min_diff = float('inf')
    best_eer = float('inf')
    
    for threshold in thresholds:
        predictions = (scores <= threshold).astype(int)
        
        # FAR, FRR 계산 (division by zero 방지)
        legitimate_count = np.sum(labels == 1)
        impostor_count = np.sum(labels == 0)
        
        if impostor_count == 0 or legitimate_count == 0:
            continue
            
        far = np.sum((predictions == 1) & (labels == 0)) / impostor_count
        frr = np.sum((predictions == 0) & (labels == 1)) / legitimate_count
        
        # EER 업데이트 (올바른 로직)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_eer = (far + frr) / 2
    
    return best_eer * 100  # 백분율로 변환

def train_model():
    """모델 훈련"""
    # 설정 로딩
    config = Config()
    hyperparams = config.get_config_dict()["hyperparams"]
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() and config.get_config_dict()["GPU"] == "True" else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로딩
    training_data, validation_data = load_data()
    
    # 모델 생성
    # 성능 최적화 설정
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    model = create_model(config)
    model = model.to(device)
    
    # 손실 함수 선택
    loss_type = "set2set"  # 또는 "triplet"
    if loss_type == "triplet":
        loss_fn = get_triplet_loss(
            loss_type='standard',
            margin=hyperparams["margin"],
            distance_metric='euclidean'
        )
    else:
        loss_fn = get_set2set_loss(
            loss_type='simplified',
            margin=hyperparams["margin"],
            beta=0.05
        )
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=1e-4)

    # 스케줄러: 5 epoch 워밍업 + 코사인
    warmup_epochs = 5
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400 - warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    
    # 데이터 생성기
    curriculum_generator = CurriculumGenerator(
        training_data,
        seq_len=config.get_config_dict()["data"]["keystroke_sequence_len"],
        curriculum_delay=10
    )
    
    # 훈련 설정
    epochs = 400
    batch_size = hyperparams["batch_size"]["aalto"]
    epoch_batch_count = hyperparams["epoch_batch_count"]["aalto"]
    
    # 저장 경로
    save_path = "models"
    os.makedirs(save_path, exist_ok=True)
    
    # 콜백
    callback = TrainingCallback(model, optimizer, save_path, patience=100)
    
    print("Starting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {hyperparams['learning_rate']}")
    print(f"Loss type: {loss_type}")
    
    # 훈련 루프
    scaler = GradScaler(device.type if device.type in ('cuda', 'cpu') else 'cpu', enabled=(device.type == 'cuda'))

    for epoch in range(epochs):
        model.train()
        
        # 커리큘럼 생성기 업데이트
        curriculum_generator.on_epoch_begin(epoch)
        
        # 배치 생성
        data_generator = curriculum_generator(batch_size)
        
        epoch_loss = 0.0
        batch_count = 0
        
        start_time = time.time()
        
        # 에포크별 배치 처리
        for _ in range(epoch_batch_count):
            try:
                anchor, positive, negative = next(data_generator)
                
                # GPU로 이동
                anchor = torch.as_tensor(anchor, dtype=torch.float32, device=device)
                positive = torch.as_tensor(positive, dtype=torch.float32, device=device)
                negative = torch.as_tensor(negative, dtype=torch.float32, device=device)
                
                # 순전파 (AMP)
                with autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.bfloat16):
                    anchor_emb = model(anchor)
                    positive_emb = model(positive)
                    negative_emb = model(negative)
                
                # 손실 계산
                if loss_type == "triplet":
                    loss = loss_fn(anchor_emb, positive_emb, negative_emb)
                else:
                    loss = loss_fn(anchor_emb, positive_emb, negative_emb)
                
                # 역전파
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # AMP 사용 시 gradient clipping은 unscale 후 적용해야 함
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                batch_count += 1
                
            except StopIteration:
                break
        
        # 평균 손실 계산
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0

        # 스케줄러 스텝
        scheduler.step()
        
        # 검증
        if epoch % 2 == 0:  # 2 에포크마다 검증 (디버깅용)
            eer = evaluate_model(model, validation_data, config, device)
        else:
            eer = float('inf')  # 검증하지 않는 에포크
        
        # 콜백 처리
        should_stop = callback.on_epoch_end(epoch, avg_loss, eer)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Time: {epoch_time:.2f}s")
        
        if should_stop:
            print("Training stopped due to early stopping")
            break
    
    print("Training completed!")
    print(f"Best EER: {callback.best_eer:.4f}")

if __name__ == "__main__":
    train_model() 