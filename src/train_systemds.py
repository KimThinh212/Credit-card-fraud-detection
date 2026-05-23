"""
Huấn luyện mô hình Credit Card Fraud Detection sử dụng Apache SystemDS
Chức năng:
  - Đọc dữ liệu Train đã chuẩn hóa từ data/processed/
  - Huấn luyện mô hình Logistic Regression (multiLogReg) với SystemDS
  - Lưu ma trận trọng số (đã điều chỉnh dấu) vào data/processed/model_weights.csv
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from systemds.context import SystemDSContext
from systemds.operator.algorithm import multiLogReg


def read_dataframe(data_path):
    df = pd.read_csv(data_path)
    values = df.values.astype(np.float64)
    logger.info(f"Đã đọc {data_path}: shape = {values.shape}")
    return values


def train_model():
    """
    Huấn luyện mô hình sử dụng SystemDS multiLogReg (Multinomial Logistic Regression).

    Quy trình:
      1. Đọc X_train, y_train từ CSV
      2. Chuyển nhãn {0, 1} sang {1, 2} (yêu cầu của multiLogReg)
      3. Nạp vào SystemDS dưới dạng matrix
      4. Huấn luyện bằng multiLogReg
      5. Tách bias (phần tử cuối) và feature weights
      6. Nghịch đảo dấu weights để tương thích với API predict
      7. Lưu vào model_weights.csv
    """
    data_dir = os.path.join(project_root, "data", "processed")
    X_train_path = os.path.join(data_dir, "X_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    output_path = os.path.join(data_dir, "model_weights.csv")

    for path in [X_train_path, y_train_path]:
        if not os.path.exists(path):
            logger.error(f"Không tìm thấy file: {path}")
            logger.error("Hãy chạy data_processing.py trước.")
            sys.exit(1)

    logger.info("Đang đọc dữ liệu Train...")
    X_np = read_dataframe(X_train_path)
    y_np = read_dataframe(y_train_path)

    n_samples, n_features = X_np.shape
    logger.info(f"Số lượng mẫu: {n_samples}, Số lượng features: {n_features}")

    if y_np.ndim > 1 and y_np.shape[1] == 1:
        y_np = y_np.flatten()

    # multiLogReg yêu cầu labels trong encoding {1, 2}
    # {0: legit, 1: fraud} -> {1: legit, 2: fraud}
    y_np = y_np + 1.0

    with SystemDSContext() as ctx:
        logger.info("Đã khởi tạo SystemDSContext.")

        X = ctx.from_numpy(X_np)
        y = ctx.from_numpy(y_np)

        logger.info("Đang huấn luyện mô hình với multiLogReg...")
        logger.info("  -> Thuật toán: multiLogReg (Multinomial Logistic Regression)")
        logger.info("  -> reg=0.001 (hệ số regularization)")

        weights_matrix = multiLogReg(X, y, reg=0.001)
        w_full = weights_matrix.compute().flatten()

        # weights có shape (n_features + 1,) với bias ở vị trí CUỐI CÙNG
        # multiLogReg học log(P(Y=2)/P(Y=1)) = X@w + b
        # Ta cần nghịch đảo dấu để API predict chuẩn (sigmoid >= 0.5)
        raw_weights = w_full[:-1]
        raw_bias = w_full[-1]

        weights = -raw_weights
        bias = -raw_bias

        logger.info(f"Huấn luyện hoàn tất!")
        logger.info(f"Kích thước weights: {weights.shape}")
        logger.info(f"Bias (intercept): {bias:.6f}")

        feature_names = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]

        weights_df = pd.DataFrame({
            "feature": feature_names,
            "weight": weights
        })

        bias_entry = pd.DataFrame([{"feature": "bias", "weight": float(bias)}])
        weights_df = pd.concat([weights_df, bias_entry], ignore_index=True)

        weights_df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu model weights vào {output_path}")

        weights_abs = weights_df.iloc[:-1].copy()
        weights_abs["abs_weight"] = weights_abs["weight"].abs()
        top_features = weights_abs.nlargest(10, "abs_weight")
        logger.info("Top 10 features quan trọng nhất:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['weight']:.6f}")

        return weights, bias


if __name__ == "__main__":
    train_model()
