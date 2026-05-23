import os
import sys
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
    classification_report
)

REPORTS_DIR = os.path.join(project_root, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

np.random.seed(42)

print("=" * 60)
print("ĐÁNH GIÁ MÔ HÌNH CREDIT CARD FRAUD DETECTION")
print("=" * 60)

X_test = pd.read_csv(os.path.join(project_root, "data", "processed", "X_test.csv")).values
y_test = pd.read_csv(os.path.join(project_root, "data", "processed", "y_test.csv")).values.ravel()

model_df = pd.read_csv(os.path.join(project_root, "data", "processed", "model_weights.csv"))
weights = model_df[model_df['feature'] != 'bias']['weight'].values.astype(np.float64)
bias = model_df[model_df['feature'] == 'bias']['weight'].values[0]

print(f"\n📊 Dữ liệu Test:  {X_test.shape[0]} mẫu, {X_test.shape[1]} features")
print(f"📊 Test - Legit: {(y_test==0).sum()}, Fraud: {(y_test==1).sum()}")
print(f"📊 Model - Bias: {bias:.4f}, Features: {len(weights)}")

logits = X_test @ weights + bias
y_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*60}")
print("KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET")
print(f"{'='*60}")
print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision: {prec:.4f}  ({prec*100:.2f}%)")
print(f"  Recall   : {rec:.4f}  ({rec*100:.2f}%)")
print(f"  F1-score : {f1:.4f}  ({f1*100:.2f}%)")
print(f"  ROC-AUC  : {roc_auc:.4f}  ({roc_auc*100:.2f}%)")
print(f"{'='*60}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legit (0)', 'Fraud (1)']))

cm = confusion_matrix(y_test, y_pred)
print(f"📋 Confusion Matrix:")
print(f"              Predicted")
print(f"              Legit  Fraud")
print(f"  Actual Legit  {cm[0,0]:3d}    {cm[0,1]:3d}")
print(f"         Fraud  {cm[1,0]:3d}    {cm[1,1]:3d}")

# ====== 1. Confusion Matrix ======
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Legit', 'Fraud'],
    cmap='Blues',
    ax=ax,
    colorbar=False,
    values_format='d'
)
ax.set_title('Confusion Matrix - Credit Card Fraud Detection (SystemDS multiLogReg)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Đã lưu: reports/confusion_matrix.png")

# ====== 2. ROC Curve ======
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#2563EB', linewidth=2.5, label=f'SystemDS multiLogReg (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='#9CA3AF', linestyle='--', linewidth=1.5, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.15, color='#2563EB')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curve - Credit Card Fraud Detection', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Đã lưu: reports/roc_curve.png")

# ====== 3. Precision-Recall Curve ======
precision, recall, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ap = np.trapz(precision, recall)
ax.plot(recall, precision, color='#10B981', linewidth=2.5, label=f'SystemDS multiLogReg (AP = {ap:.4f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve - Credit Card Fraud Detection', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'precision_recall_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Đã lưu: reports/precision_recall_curve.png")

# ====== 4. Feature Importance ======
feature_names = [f'V{i}' for i in range(1, 29)] + ['Time_scaled', 'Amount_scaled']
feat_imp = pd.DataFrame({'feature': feature_names, 'weight': weights, 'abs_weight': np.abs(weights)})
feat_imp = feat_imp.sort_values('abs_weight', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#EF4444' if w > 0 else '#3B82F6' for w in feat_imp['weight'].values]
ax.barh(range(len(feat_imp)), feat_imp['weight'].values, color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(feat_imp)))
ax.set_yticklabels(feat_imp['feature'].values, fontsize=11)
ax.set_xlabel('Weight Coefficient', fontsize=12)
ax.set_title('Top 15 Feature Weights - SystemDS multiLogReg', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='#9CA3AF', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')
for i, (v, w) in enumerate(zip(feat_imp['feature'], feat_imp['weight'])):
    ax.text(w + (0.01 if w >= 0 else -0.01), i, f'{w:.4f}',
            va='center', ha='left' if w >= 0 else 'right', fontsize=9, color='#374151')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Đã lưu: reports/feature_importance.png")

# ====== 5. Probability Distribution ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax_idx, (condition, label, color) in enumerate([
    (y_test == 0, 'Legit Transactions', '#10B981'),
    (y_test == 1, 'Fraud Transactions', '#EF4444'),
]):
    ax = axes[ax_idx]
    ax.hist(y_prob[condition], bins=20, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvline(x=0.5, color='#374151', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
    ax.set_xlabel('Predicted Fraud Probability', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{label} (n={condition.sum()})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
plt.suptitle('Probability Distribution by Actual Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'probability_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Đã lưu: reports/probability_distribution.png")

# ====== 6. Metrics Summary Bar Chart ======
metrics_dict = {
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1-Score': f1,
    'ROC-AUC': roc_auc,
}
fig, ax = plt.subplots(figsize=(9, 5))
colors_metrics = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EF4444']
bars = ax.bar(range(len(metrics_dict)), list(metrics_dict.values()), color=colors_metrics, edgecolor='white', linewidth=0.5, width=0.6)
ax.set_xticks(range(len(metrics_dict)))
ax.set_xticklabels(list(metrics_dict.keys()), fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Metrics - SystemDS multiLogReg', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, metrics_dict.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'metrics_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Đã lưu: reports/metrics_summary.png")

print(f"\n{'='*60}")
print(f"🎉 HOÀN TẤT! Tất cả biểu đồ đánh giá đã được lưu trong: {REPORTS_DIR}")
print(f"{'='*60}")
print(f"\n📁 Các file đã tạo:")
for f in sorted(os.listdir(REPORTS_DIR)):
    fpath = os.path.join(REPORTS_DIR, f)
    size = os.path.getsize(fpath)
    print(f"   📊 {f} ({size/1024:.1f} KB)")
