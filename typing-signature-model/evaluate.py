import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import numpy as np
import pickle
import os
import argparse
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    average_precision_score, balanced_accuracy_score, roc_curve
)

from utils.config import Config

class TestDataset:
    """테스트/검증용 데이터셋 (평가 전용, 최소 의존성)"""
    def __init__(self, eval_data, seq_len: int = 50):
        self.eval_data = eval_data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.eval_data) * len(self.eval_data[0])

    def __getitem__(self, idx):
        user_idx = idx // len(self.eval_data[0])
        session_idx = idx % len(self.eval_data[0])
        sequence = self.eval_data[user_idx][session_idx]
        return self.pad_sequence(sequence)

    def pad_sequence(self, sequence):
        if len(sequence) == self.seq_len:
            return sequence
        elif len(sequence) < self.seq_len:
            padding = np.zeros((self.seq_len - len(sequence), 3))
            return np.vstack([sequence, padding])
        else:
            return sequence[:self.seq_len]


class ModelEvaluator:
    """모델 평가 클래스"""

    def __init__(self, model_path, config, device):
        self.model_path = model_path
        self.config = config
        self.device = device

        # 모델 로딩 (PyTorch 2.6+ 기본 weights_only=True로 변경 → 전체 객체 로드 허용)
        self.model = torch.load(model_path, map_location=device, weights_only=False)
        self.model.eval()

        # 테스트 데이터 로딩
        with open("data/testing_data.pickle", 'rb') as f:
            self.testing_data = pickle.load(f)

        print(f"Model loaded from: {model_path}")
        print(f"Testing users: {len(self.testing_data)}")

    def extract_embeddings(self, data):
        """임베딩 추출"""
        seq_len = self.config.get_config_dict()["data"]["keystroke_sequence_len"]
        batch_size = self.config.get_config_dict()["hyperparams"]["batch_size"]["aalto"]

        dataset = TestDataset(data, seq_len=seq_len)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.float().to(self.device, non_blocking=True)
                with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda'), dtype=torch.float16):
                    batch_embeddings = self.model(batch)
                embeddings.append(batch_embeddings.detach().cpu())

        return torch.cat(embeddings, dim=0)

    # ---- EER 계산: 기존 스윕 방식 유지 (distance 기준, 낮을수록 양성) ----
    def calculate_eer(self, scores, labels):
        """EER (Equal Error Rate) 계산 - distance 기준 스위프"""
        thresholds = np.linspace(scores.min(), scores.max(), 2048)
        best_eer = float('inf')
        best_threshold = 0.0
        min_diff = float('inf')

        # 분모 0 방지
        num_neg = max(1, np.sum(labels == 0))
        num_pos = max(1, np.sum(labels == 1))

        for thr in thresholds:
            # distance <= thr => positive(=genuine)
            pred = (scores <= thr).astype(int)

            far = np.sum((pred == 1) & (labels == 0)) / num_neg  # FPR
            frr = np.sum((pred == 0) & (labels == 1)) / num_pos  # FNR

            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                best_eer = (far + frr) / 2.0
                best_threshold = thr

        return best_eer * 100.0, best_threshold

    # ---- EER 계산: ROC 기반(권장) ----
    def calculate_eer_via_roc(self, scores, labels):
        """
        EER (Equal Error Rate) 계산 - ROC 기반 보간 (권장)
        내부적으로 similarity = -distance 로 변환 후 ROC 계산
        반환 threshold는 distance 기준으로 되돌려서 반환
        """
        sim = -scores  # <<< FIX: 방향 통일(클수록 양성)
        fpr, tpr, thresholds = roc_curve(labels, sim)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fnr[idx] + fpr[idx]) / 2.0 * 100.0
        best_sim_thr = thresholds[idx]
        best_dist_thr = -best_sim_thr  # similarity -> distance 역변환
        return eer, best_dist_thr

    def calculate_roc_auc(self, scores, labels):
        """ROC-AUC 계산 (distance -> similarity 변환)"""
        sim = -scores  # <<< FIX: distance를 similarity로
        return roc_auc_score(labels, sim)

    def calculate_accuracy(self, scores, labels, threshold):
        """정확도 계산 (distance 기준 임계값)"""
        predictions = (scores <= threshold).astype(int)
        return accuracy_score(labels, predictions)

    def calculate_precision_recall(self, scores, labels, threshold):
        """정밀도와 재현율 계산 (distance 기준 임계값)"""
        predictions = (scores <= threshold).astype(int)
        precision = precision_score(labels, predictions, zero_division=0)  # <<< FIX: 0분모 안전
        recall = recall_score(labels, predictions, zero_division=0)
        return precision, recall

    def evaluate_user(self, user_idx, enroll_sessions=10, verify_sessions=5):
        """사용자별 평가 (1:N 검증)"""
        user_embeddings = self.user_embeddings[user_idx]

        # 등록 임베딩 (평균)
        enroll_emb = torch.mean(user_embeddings[:enroll_sessions], dim=0)

        scores, labels = [], []

        # 합법 사용자 검증 (genuine = 1)
        for session_idx in range(enroll_sessions, enroll_sessions + verify_sessions):
            verify_emb = user_embeddings[session_idx]
            distance = torch.norm(enroll_emb - verify_emb)
            scores.append(distance.item())
            labels.append(1)

        # 불법 사용자 검증 (impostor = 0)
        num_users = len(self.testing_data)
        for other_user in range(num_users):
            if other_user == user_idx:
                continue
            for session_idx in range(enroll_sessions, enroll_sessions + verify_sessions):
                verify_emb = self.user_embeddings[other_user, session_idx]
                distance = torch.norm(enroll_emb - verify_emb)
                scores.append(distance.item())
                labels.append(0)

        return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)

    def evaluate_all_users(self):
        """모든 사용자 평가"""
        print("Extracting embeddings...")
        self.user_embeddings = self.extract_embeddings(self.testing_data)

        # 사용자별 임베딩 재구성 [num_users, num_sessions, dim]
        num_users = len(self.testing_data)
        num_sessions = len(self.testing_data[0])
        self.user_embeddings = self.user_embeddings.view(num_users, num_sessions, -1)

        print("Evaluating all users...")

        all_scores = []
        all_labels = []

        enroll_sessions = self.config.get_config_dict()["hyperparams"]["number_of_enrollment_sessions"]["aalto"]
        verify_sessions = self.config.get_config_dict()["hyperparams"]["number_of_verify_sessions"]["aalto"]

        for user_idx in range(num_users):
            scores, labels = self.evaluate_user(user_idx, enroll_sessions, verify_sessions)
            all_scores.extend(scores)
            all_labels.extend(labels)

        all_scores = np.array(all_scores, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int32)

        # ---- 성능 지표 계산 ----
        # 권장: ROC 기반 EER 사용 (threshold 일관성 distance 기준)
        eer, threshold = self.calculate_eer_via_roc(all_scores, all_labels)

        roc_auc = self.calculate_roc_auc(all_scores, all_labels)              # <<< FIX 반영
        accuracy = self.calculate_accuracy(all_scores, all_labels, threshold)
        precision, recall = self.calculate_precision_recall(all_scores, all_labels, threshold)

        # 불균형 지표 추가
        pr_auc = average_precision_score(all_labels, -all_scores)             # similarity = -distance
        bal_acc = balanced_accuracy_score(all_labels, (all_scores <= threshold).astype(int))

        return {
            'EER': eer,
            'ROC_AUC': roc_auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'PR_AUC': pr_auc,
            'Balanced_Accuracy': bal_acc,
            'Threshold': threshold,
            'Scores': all_scores,
            'Labels': all_labels
        }

    def save_results(self, results, save_path="results"):
        """결과 출력 (파일 저장 의존성 최소화)"""

        # 결과 출력
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"EER: {results['EER']:.4f}%")
        print(f"ROC-AUC: {results['ROC_AUC']:.4f}")
        print(f"PR-AUC: {results['PR_AUC']:.4f}")
        print(f"Balanced-Accuracy: {results['Balanced_Accuracy']:.4f}")
        print(f"Accuracy: {results['Accuracy']:.4f}")
        print(f"Precision: {results['Precision']:.4f}")
        print(f"Recall: {results['Recall']:.4f}")
        print(f"Threshold (distance): {results['Threshold']:.4f}")
        print("=" * 50)



def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Evaluate keystroke authentication model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')

    args = parser.parse_args()

    # 설정 로딩
    config = Config()

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    # 평가 실행
    evaluator = ModelEvaluator(args.model_path, config, device)
    results = evaluator.evaluate_all_users()
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
