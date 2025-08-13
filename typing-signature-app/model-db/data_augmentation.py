import numpy as np
import random
from typing import List, Tuple
import torch

class DataAugmentation:
    """키스트로크 데이터 증강 클래스"""
    
    def __init__(self, noise_ratio=0.05, mix_ratio=0.3):
        self.noise_ratio = noise_ratio
        self.mix_ratio = mix_ratio
    
    def noise_injection(self, sequence: np.ndarray) -> np.ndarray:
        """입력 피처에 ±5~10% noise 주입"""
        noise = np.random.normal(0, self.noise_ratio, sequence.shape)
        augmented_sequence = sequence + noise * sequence
        return augmented_sequence
    
    def synthetic_user_mixing(self, user1_sessions: List[np.ndarray], 
                            user2_sessions: List[np.ndarray]) -> List[np.ndarray]:
        """유사 사용자 벡터 평균을 기반으로 mix"""
        mixed_sessions = []
        
        for i in range(min(len(user1_sessions), len(user2_sessions))):
            # 두 사용자의 세션을 가중 평균
            mix_weight = np.random.uniform(0.3, 0.7)
            mixed_session = (mix_weight * user1_sessions[i] + 
                           (1 - mix_weight) * user2_sessions[i])
            mixed_sessions.append(mixed_session)
        
        return mixed_sessions
    
    def curriculum_hard_sampling(self, all_users: List[List[np.ndarray]], 
                               epoch: int, max_epochs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """epoch에 따라 더 어려운 (유사 사용자) 샘플 선택"""
        
        # epoch에 따라 어려운 샘플 비율 증가
        difficulty_ratio = epoch / max_epochs
        
        # 사용자 선택
        user1_idx = np.random.randint(0, len(all_users))
        user2_idx = np.random.randint(0, len(all_users))
        
        # 어려운 샘플링: 유사한 사용자 선택 확률 증가
        if np.random.random() < difficulty_ratio:
            # 유사한 사용자 찾기 (여기서는 간단히 랜덤 선택)
            while user2_idx == user1_idx:
                user2_idx = np.random.randint(0, len(all_users))
        
        # 세션 선택
        session1_idx = np.random.randint(0, len(all_users[user1_idx]))
        session2_idx = np.random.randint(0, len(all_users[user2_idx]))
        
        # 앵커, 포지티브, 네거티브 생성
        anchor = all_users[user1_idx][session1_idx]
        positive = all_users[user1_idx][session2_idx]
        negative = all_users[user2_idx][session2_idx]
        
        return anchor, positive, negative

class CurriculumGenerator:
    """커리큘럼 학습을 위한 데이터 생성기"""
    
    def __init__(self, training_data: List[List[np.ndarray]], 
                 seq_len: int = 50, curriculum_delay: int = 20):
        self.training_data = training_data
        self.seq_len = seq_len
        self.curriculum_delay = curriculum_delay
        self.epoch = 0
        self.augmenter = DataAugmentation()
    
    def pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """시퀀스를 일정 길이로 패딩"""
        if len(sequence) == self.seq_len:
            return sequence
        elif len(sequence) < self.seq_len:
            # 0으로 패딩
            padding = np.zeros((self.seq_len - len(sequence), 3))
            return np.vstack([sequence, padding])
        else:
            return sequence[:self.seq_len]
    
    def get_triplet_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """트리플렛 배치 생성"""
        anchors = []
        positives = []
        negatives = []
        
        for _ in range(batch_size):
            # 사용자 선택
            user1_idx = np.random.randint(0, len(self.training_data))
            user2_idx = np.random.randint(0, len(self.training_data))
            
            while user2_idx == user1_idx:
                user2_idx = np.random.randint(0, len(self.training_data))
            
            # 세션 선택
            session1_idx = np.random.randint(0, len(self.training_data[user1_idx]))
            session2_idx = np.random.randint(0, len(self.training_data[user1_idx]))
            session3_idx = np.random.randint(0, len(self.training_data[user2_idx]))
            
            # 데이터 증강 적용
            anchor = self.augmenter.noise_injection(
                self.pad_sequence(self.training_data[user1_idx][session1_idx])
            )
            positive = self.augmenter.noise_injection(
                self.pad_sequence(self.training_data[user1_idx][session2_idx])
            )
            negative = self.augmenter.noise_injection(
                self.pad_sequence(self.training_data[user2_idx][session3_idx])
            )
            
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
        
        return (np.array(anchors), np.array(positives), np.array(negatives))
    
    def on_epoch_begin(self, epoch: int):
        """에포크 시작 시 호출"""
        self.epoch = epoch
        print(f"[CurriculumGenerator] EPOCH {epoch}")
        
        if epoch < self.curriculum_delay:
            print("[CurriculumGenerator] Basic training without curriculum")
        else:
            print("[CurriculumGenerator] Curriculum training with hard sampling")
    
    def __call__(self, batch_size: int):
        """데이터 생성기"""
        while True:
            if self.epoch < self.curriculum_delay:
                # 기본 트리플렛 생성
                yield self.get_triplet_batch(batch_size)
            else:
                # 커리큘럼 하드 샘플링
                difficulty_ratio = (self.epoch - self.curriculum_delay) / 100  # 최대 100 에포크
                
                anchors = []
                positives = []
                negatives = []
                
                for _ in range(batch_size):
                    anchor, positive, negative = self.augmenter.curriculum_hard_sampling(
                        self.training_data, self.epoch, 100
                    )
                    
                    # 데이터 증강 적용
                    anchor = self.augmenter.noise_injection(self.pad_sequence(anchor))
                    positive = self.augmenter.noise_injection(self.pad_sequence(positive))
                    negative = self.augmenter.noise_injection(self.pad_sequence(negative))
                    
                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)
                
                yield (np.array(anchors), np.array(positives), np.array(negatives))

class TestDataset:
    """테스트/검증용 데이터셋"""
    
    def __init__(self, eval_data: List[List[np.ndarray]], seq_len: int = 50):
        self.eval_data = eval_data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.eval_data) * len(self.eval_data[0])
    
    def __getitem__(self, idx):
        user_idx = idx // len(self.eval_data[0])
        session_idx = idx % len(self.eval_data[0])
        
        sequence = self.eval_data[user_idx][session_idx]
        return self.pad_sequence(sequence)
    
    def pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """시퀀스를 일정 길이로 패딩"""
        if len(sequence) == self.seq_len:
            return sequence
        elif len(sequence) < self.seq_len:
            padding = np.zeros((self.seq_len - len(sequence), 3))
            return np.vstack([sequence, padding])
        else:
            return sequence[:self.seq_len] 