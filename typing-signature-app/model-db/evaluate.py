import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
import numpy as np
import pickle
import os
import sys
import argparse
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    average_precision_score, balanced_accuracy_score, roc_curve,
    precision_recall_curve, confusion_matrix
)
import matplotlib.pyplot as plt

# 경로 설정 (프로젝트 루트의 utils 접근)
sys.path.append(str((Path(__file__).resolve().parent / "utils").resolve()))

from config import Config
from model import create_model
from data_augmentation import TestDataset


# -----------------------------
# Utility: score/metric helpers
# -----------------------------
def to_similarity(distance_scores: np.ndarray) -> np.ndarray:
    """distance(작을수록 genuine) -> similarity(클수록 genuine)"""
    return -distance_scores

def eval_at_threshold(scores_dist: np.ndarray, labels: np.ndarray, thr: float):
    """distance 기준 임계값으로 평가"""
    preds = (scores_dist <= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    far = fp / (fp + tn + 1e-12)
    frr = fn / (tp + fn + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    acc  = (tp + tn) / (tp + tn + fp + fn)
    return dict(Threshold=thr, FAR=far, FRR=frr, Precision=prec, Recall=rec, F1=f1, Accuracy=acc)

def pick_thresholds(scores_dist: np.ndarray, labels: np.ndarray):
    """
    운영에 필요한 임계값들을 한 번에 계산:
    - FAR@0.1%
    - FAR@0.01%
    - F1-max (PR 곡선 기반)
    반환 임계값은 모두 distance 기준(threshold 작을수록 엄격)
    """
    sim = to_similarity(scores_dist)
    fpr, tpr, thr_sim_roc = roc_curve(labels, sim)
    precisions, recalls, thr_sim_pr = precision_recall_curve(labels, sim)

    # FAR 타깃 임계값 (ROC 기반)
    def thr_for_far(target_far: float) -> float:
        idx = np.searchsorted(fpr, target_far, side='left')
        idx = min(idx, len(thr_sim_roc) - 1)
        return -thr_sim_roc[idx]  # sim -> dist

    thr_far_01   = thr_for_far(0.001)   # FAR 0.1%
    thr_far_001  = thr_for_far(0.0001)  # FAR 0.01%

    # F1-max 임계값 (PR 기반)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best = int(np.nanargmax(f1s))
    thr_sim_for_f1 = thr_sim_pr[max(min(best, len(thr_sim_pr)-1), 0)]
    thr_f1 = -thr_sim_for_f1

    return {
        "FAR@0.1%": thr_far_01,
        "FAR@0.01%": thr_far_001,
        "F1-max": thr_f1
    }

def decide_3way(score_dist: float, thr_low: float, thr_high: float):
    """
    3단계 판정:
    - score <= thr_low  : ACCEPT(자동 승인)
    - thr_low < score <= thr_high : REVERIFY(추가 인증)
    - score > thr_high  : REJECT(거부)
    """
    if score_dist <= thr_low:
        return "ACCEPT"
    elif score_dist <= thr_high:
        return "REVERIFY"
    else:
        return "REJECT"

def l2_normalize(x: torch.Tensor, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = l2_normalize(a); b = l2_normalize(b)
    sim = (a * b).sum(dim=-1)
    return 1.0 - sim  # distance


class ModelEvaluator:
    """모델 평가 클래스"""

    def __init__(self, model_path, config, device,
                 distance_metric: str = "l2",
                 template_method: str = "mean",
                 znorm_enabled: bool = False,
                 cohort_size: int = 30,
                 rng_seed: int = 42):
        self.model_path = model_path
        self.config = config
        self.device = device

        self.distance_metric = distance_metric  # 'l2' | 'cosine'
        self.template_method = template_method  # 'mean' | 'median' | 'robust'
        self.znorm_enabled   = znorm_enabled
        self.cohort_size     = max(1, int(cohort_size))
        self.rng = np.random.RandomState(rng_seed)

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
                with autocast('cuda', enabled=(self.device.type == 'cuda'), dtype=torch.bfloat16):
                    batch_embeddings = self.model(batch)
                embeddings.append(batch_embeddings.detach().cpu())

        return torch.cat(embeddings, dim=0)

    # ---- 템플릿 계산 ----
    def _make_template(self, vectors: torch.Tensor) -> torch.Tensor:
        if self.template_method == "median":
            return torch.median(vectors, dim=0).values
        elif self.template_method == "robust":
            # IQR 기반 outlier 제거 후 평균
            q1 = torch.quantile(vectors, 0.25, dim=0)
            q3 = torch.quantile(vectors, 0.75, dim=0)
            iqr = q3 - q1
            mask = ((vectors >= (q1 - 1.5 * iqr)) & (vectors <= (q3 + 1.5 * iqr))).all(dim=1)
            kept = vectors[mask]
            return kept.mean(dim=0) if kept.size(0) > 0 else vectors.mean(dim=0)
        # default mean
        return vectors.mean(dim=0)

    # ---- 거리 계산 ----
    def _compute_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "cosine":
            return cosine_distance(a, b)
        return torch.norm(a - b)  # L2

    # ---- Z-Norm (사용자별 cohort 기반) ----
    def _apply_znorm(self, scores_list, user_idx, enroll_emb):
        num_users = len(self.testing_data)
        # 다른 사용자 중 무작위 cohort_size명 선정, 각 1세션 샘플
        candidates = [i for i in range(num_users) if i != user_idx]
        self.rng.shuffle(candidates)
        cohort_users = candidates[:self.cohort_size]

        cohort_dists = []
        for u in cohort_users:
            # 첫 세션만 사용 (원하면 랜덤 세션으로 바꿔도 됨)
            vec = self.user_embeddings[u, 0]
            d = float(self._compute_distance(enroll_emb, vec))
            cohort_dists.append(d)
        cohort_dists = np.array(cohort_dists, dtype=np.float64)

        mu, sigma = cohort_dists.mean(), cohort_dists.std() + 1e-12
        znormed = [(s - mu) / sigma for s in scores_list]
        return znormed

    # ---- EER 계산: 기존 스윕 방식 유지 (distance 기준, 낮을수록 양성) ----
    def calculate_eer(self, scores, labels):
        thresholds = np.linspace(scores.min(), scores.max(), 2048)
        best_eer = float('inf')
        best_threshold = 0.0
        num_neg = max(1, np.sum(labels == 0))
        num_pos = max(1, np.sum(labels == 1))
        for thr in thresholds:
            pred = (scores <= thr).astype(int)
            far = np.sum((pred == 1) & (labels == 0)) / num_neg
            frr = np.sum((pred == 0) & (labels == 1)) / num_pos
            eer_here = (far + frr) / 2.0
            if abs(far - frr) < abs(best_eer - eer_here):
                best_eer = eer_here
                best_threshold = thr
        return best_eer * 100.0, best_threshold

    # ---- EER 계산: ROC 기반(권장) ----
    def calculate_eer_via_roc(self, scores, labels):
        sim = to_similarity(scores)
        fpr, tpr, thresholds = roc_curve(labels, sim)
        fnr = 1 - tpr
        idx = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = (fnr[idx] + fpr[idx]) / 2.0 * 100.0
        best_sim_thr = thresholds[idx]
        best_dist_thr = -best_sim_thr
        return eer, best_dist_thr

    def calculate_roc_auc(self, scores, labels):
        sim = to_similarity(scores)
        return roc_auc_score(labels, sim)

    def calculate_accuracy(self, scores, labels, threshold):
        predictions = (scores <= threshold).astype(int)
        return accuracy_score(labels, predictions)

    def calculate_precision_recall(self, scores, labels, threshold):
        predictions = (scores <= threshold).astype(int)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        return precision, recall

    def evaluate_user(self, user_idx, enroll_sessions=10, verify_sessions=5):
        """사용자별 평가 (1:N 검증)"""
        user_embeddings = self.user_embeddings[user_idx]

        # 등록 임베딩
        enroll_emb = self._make_template(user_embeddings[:enroll_sessions])

        scores, labels = [], []

        # Genuine
        for session_idx in range(enroll_sessions, enroll_sessions + verify_sessions):
            verify_emb = user_embeddings[session_idx]
            distance = self._compute_distance(enroll_emb, verify_emb)
            scores.append(float(distance))
            labels.append(1)

        # Impostor
        num_users = len(self.testing_data)
        for other_user in range(num_users):
            if other_user == user_idx:
                continue
            for session_idx in range(enroll_sessions, enroll_sessions + verify_sessions):
                verify_emb = self.user_embeddings[other_user, session_idx]
                distance = self._compute_distance(enroll_emb, verify_emb)
                scores.append(float(distance))
                labels.append(0)

        # Z-Norm (옵션)
        if self.znorm_enabled:
            scores = self._apply_znorm(scores, user_idx, enroll_emb)

        return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)

    def evaluate_all_users(self):
        """모든 사용자 평가"""
        print("Extracting embeddings...")
        self.user_embeddings = self.extract_embeddings(self.testing_data)

        # [num_users, num_sessions, dim]
        num_users = len(self.testing_data)
        num_sessions = len(self.testing_data[0])
        self.user_embeddings = self.user_embeddings.view(num_users, num_sessions, -1)

        print("Evaluating all users...")

        all_scores, all_labels = [], []
        hp = self.config.get_config_dict()["hyperparams"]
        enroll_sessions = hp["number_of_enrollment_sessions"]["aalto"]
        verify_sessions = hp["number_of_verify_sessions"]["aalto"]

        for user_idx in range(num_users):
            s, l = self.evaluate_user(user_idx, enroll_sessions, verify_sessions)
            all_scores.extend(s)
            all_labels.extend(l)

        all_scores = np.array(all_scores, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int32)

        # ---- 성능 지표 ----
        eer, threshold = self.calculate_eer_via_roc(all_scores, all_labels)
        roc_auc = self.calculate_roc_auc(all_scores, all_labels)
        accuracy = self.calculate_accuracy(all_scores, all_labels, threshold)
        precision, recall = self.calculate_precision_recall(all_scores, all_labels, threshold)
        pr_auc = average_precision_score(all_labels, to_similarity(all_scores))
        bal_acc = balanced_accuracy_score(all_labels, (all_scores <= threshold).astype(int))

        # 운영 임계값
        thresholds_ops = pick_thresholds(all_scores, all_labels)
        ops_report = {name: eval_at_threshold(all_scores, all_labels, thr)
                      for name, thr in thresholds_ops.items()}

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
            'Labels': all_labels,
            'OpsThresholds': thresholds_ops,
            'OpsReport': ops_report
        }

    def plot_roc_curve(self, scores, labels, save_path="results"):
        """ROC 곡선 플롯 (similarity = -distance)"""
        os.makedirs(save_path, exist_ok=True)
        sim = to_similarity(scores)
        fpr, tpr, _ = roc_curve(labels, sim)
        auc = roc_auc_score(labels, sim)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right"); plt.grid(True)
        plt.savefig(f"{save_path}/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_score_distribution(self, scores, labels, save_path="results"):
        """점수 분포 플롯 (distance 분포)"""
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        genuine_scores = scores[labels == 1]
        impostor_scores = scores[labels == 0]
        plt.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine')
        plt.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor')
        plt.xlabel('Distance Score'); plt.ylabel('Frequency')
        plt.title('Score Distribution (Distance)')
        plt.legend(); plt.grid(True)
        plt.savefig(f"{save_path}/score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, results, save_path="results"):
        """결과 저장"""
        os.makedirs(save_path, exist_ok=True)

        # 콘솔 출력
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
        print(f"Threshold (distance, EER): {results['Threshold']:.4f}")
        print("-" * 50)
        print("Operational thresholds (distance) & metrics:")
        for name, thr in results["OpsThresholds"].items():
            rep = results["OpsReport"][name]
            print(f"{name:>10} | thr={thr:.6f} | FAR={rep['FAR']:.6f} | FRR={rep['FRR']:.6f} | "
                  f"P={rep['Precision']:.4f} R={rep['Recall']:.4f} F1={rep['F1']:.4f} Acc={rep['Accuracy']:.4f}")
        # 가중치 운영지표 출력(선택)
        if 'Weighted' in results:
            w = results['Weighted']
            print("-" * 50)
            print(f"Weighted metrics @ target impostor ratio {w['TargetImpostorRatio']}: ")
            print(f"Acc={w['Accuracy']:.4f} | P={w['Precision']:.4f} | R={w['Recall']:.4f} | PR-AUC={w['PR_AUC']:.4f}")
        print("=" * 50)

        # CSV 저장
        import pandas as pd
        results_df = pd.DataFrame([
            {'Metric': 'EER', 'Value': results['EER'], 'Unit': '%'},
            {'Metric': 'ROC_AUC', 'Value': results['ROC_AUC'], 'Unit': ''},
            {'Metric': 'PR_AUC', 'Value': results['PR_AUC'], 'Unit': ''},
            {'Metric': 'Balanced_Accuracy', 'Value': results['Balanced_Accuracy'], 'Unit': ''},
            {'Metric': 'Accuracy', 'Value': results['Accuracy'], 'Unit': ''},
            {'Metric': 'Precision', 'Value': results['Precision'], 'Unit': ''},
            {'Metric': 'Recall', 'Value': results['Recall'], 'Unit': ''},
            {'Metric': 'Threshold(distance,EER)', 'Value': results['Threshold'], 'Unit': ''},
        ])
        results_df.to_csv(f"{save_path}/evaluation_results.csv", index=False)

        ops_rows = []
        for name, thr in results["OpsThresholds"].items():
            rep = results["OpsReport"][name]
            row = {'Mode': name, 'Threshold(distance)': thr}
            row.update(rep)
            ops_rows.append(row)
        pd.DataFrame(ops_rows).to_csv(f"{save_path}/operational_thresholds.csv", index=False)

        # 점수/라벨 저장 + 플롯
        np.save(f"{save_path}/scores.npy", results['Scores'])
        np.save(f"{save_path}/labels.npy", results['Labels'])
        self.plot_roc_curve(results['Scores'], results['Labels'], save_path)
        self.plot_score_distribution(results['Scores'], results['Labels'], save_path)
        print(f"Results saved to: {save_path}")

        # CSV 저장(추가): 가중치 운영지표
        if 'Weighted' in results:
            pd.DataFrame([results['Weighted']]).to_csv(f"{save_path}/weighted_metrics.csv", index=False)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Evaluate keystroke authentication model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--metric', type=str, default='l2', choices=['l2', 'cosine'],
                        help='Distance metric for scoring (default: l2)')
    parser.add_argument('--template', type=str, default='mean',
                        choices=['mean', 'median', 'robust'],
                        help='Enrollment template method (default: mean)')
    parser.add_argument('--znorm', action='store_true', help='Apply cohort Z-Norm')
    parser.add_argument('--cohort-size', type=int, default=30, help='Cohort size for Z-Norm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for cohort sampling')
    # 보고용: 비율 보정 메트릭
    parser.add_argument('--target-imp-ratio', type=float, default=None,
                        help='Target impostor/genuine ratio for weighted metrics (e.g., 100 for 1:100).')
    # 운영용 개인화 옵션(되돌림): 비활성화

    args = parser.parse_args()

    # 설정 로딩
    config = Config()

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    # 평가 실행
    evaluator = ModelEvaluator(
        args.model_path, config, device,
        distance_metric=('l2' if args.metric == 'l2' else 'cosine'),
        template_method=args.template,
        znorm_enabled=args.znorm,
        cohort_size=args.cohort_size,
        rng_seed=args.seed
    )
    results = evaluator.evaluate_all_users()

    # 비율 보정 메트릭(보고용)
    if args.target_imp_ratio is not None and np.isfinite(args.target_imp_ratio) and args.target_imp_ratio > 0:
        labels = results['Labels']
        scores = results['Scores']
        thr = results['Threshold']
        num_pos = max(1, np.sum(labels == 1))
        num_neg = max(1, np.sum(labels == 0))
        current_ratio = num_neg / num_pos
        neg_scale = float(args.target_imp_ratio) / current_ratio
        sample_weight = np.where(labels == 0, neg_scale, 1.0)
        from sklearn.metrics import precision_score as _precision_score, recall_score as _recall_score, accuracy_score as _accuracy_score, average_precision_score as _average_precision_score
        preds = (scores <= thr).astype(int)
        acc_w = _accuracy_score(labels, preds, sample_weight=sample_weight)
        prec_w = _precision_score(labels, preds, sample_weight=sample_weight, zero_division=0)
        rec_w = _recall_score(labels, preds, sample_weight=sample_weight, zero_division=0)
        pr_auc_w = _average_precision_score(labels, to_similarity(scores), sample_weight=sample_weight)
        results['Weighted'] = {
            'TargetImpostorRatio': float(args.target_imp_ratio),
            'Accuracy': float(acc_w),
            'Precision': float(prec_w),
            'Recall': float(rec_w),
            'PR_AUC': float(pr_auc_w)
        }

    evaluator.save_results(results)


if __name__ == "__main__":
    main()
