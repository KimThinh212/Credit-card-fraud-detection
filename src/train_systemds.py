"""
Huấn luyện mô hình Credit Card Fraud Detection sử dụng Apache SystemDS
Chức năng:
  - Đọc dữ liệu Train đã chuẩn hóa từ data/processed/
  - Huấn luyện mô hình SVM (l2svm) với SystemDS
  - Lưu ma trận trọng số vào data/processed/model_weights.csv
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
from systemds.operator.algorithm import l2svm


def read_dataframe(data_path, is_label=False):
    """
    Đọc CSV bằng pandas và trả về numpy array.
    SystemDS cần numpy array để nạp qua ctx.from_numpy().
    """
    df = pd.read_csv(data_path)
    values = df.values.astype(np.float64)
    logger.info(f"Đã đọc {data_path}: shape = {values.shape}")
    return values


def train_model():
    """
    Huấn luyện mô hình sử dụng SystemDS l2svm (binary SVM with L2 regularization).

    Quy trình:
      1. Đọc X_train, y_train từ CSV
      2. Nạp vào SystemDS dưới dạng matrix
      3. Huấn luyện bằng l2svm
      4. Tách bias và feature weights
      5. Lưu vào model_weights.csv
    """
    data_dir = os.path.join(project_root, "data", "processed")
    X_train_path = os.path.join(data_dir, "X_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    output_path = os.path.join(data_dir, "model_weights.csv")

    # Kiểm tra file đầu vào
    for path in [X_train_path, y_train_path]:
        if not os.path.exists(path):
            logger.error(f"Không tìm thấy file: {path}")
            logger.error("Hãy chạy data_processing.py trước.")
            sys.exit(1)

    # Đọc dữ liệu bằng pandas
    logger.info("Đang đọc dữ liệu Train...")
    X_np = read_dataframe(X_train_path, is_label=False)
    y_np = read_dataframe(y_train_path, is_label=True)

    n_samples, n_features = X_np.shape
    logger.info(f"Số lượng mẫu: {n_samples}, Số lượng features: {n_features}")

    # Chuyển y thành vector 1D nếu cần
    if y_np.ndim > 1 and y_np.shape[1] == 1:
        y_np = y_np.flatten()

    # l2svm yêu cầu labels trong encoding -1/+1 hoặc 1/2.
    # Dữ liệu hiện tại có Class = 0 (legit) và 1 (fraud).
    # Chuyển đổi: 0 -> -1, 1 -> +1
    y_np = np.where(y_np == 0, -1.0, 1.0)

    # Khởi tạo SystemDS Context
    with SystemDSContext() as ctx:
        logger.info("Đã khởi tạo SystemDSContext.")

        # Nạp dữ liệu vào SystemDS
        X = ctx.from_numpy(X_np)
        y = ctx.from_numpy(y_np)

        logger.info("Đang huấn luyện mô hình với l2svm...")
        logger.info("  -> Thuật toán: l2svm (L2-regularized SVM)")
        logger.info("  -> maxIterations=200 (số vòng lặp tối đa)")
        logger.info("  -> epsilon=1e-7 (ngưỡng hội tụ)")
        logger.info("  -> reg=0.001 (hệ số regularization)")
        logger.info("  -> intercept=True (thêm bias)")

        # Huấn luyện mô hình
        # l2svm trả về Matrix weights (shape: n_features+1 x 1 khi intercept=True)
        # Phần tử đầu tiên là bias, các phần tử còn lại là feature weights
        weights_matrix = l2svm(
            X, y,
            intercept=True,
            reg=0.001,
            maxIterations=200,
            epsilon=1e-7,
            verbose=False
        )

        # Lấy giá trị numpy
        weights_full = weights_matrix.compute()

        # Tách bias và feature weights
        if weights_full.ndim > 1 and weights_full.shape[1] == 1:
            weights_full = weights_full.flatten()

        # Phần tử đầu tiên là bias (vì intercept=True)
        bias = weights_full[0]
        weights = weights_full[1:]

        logger.info(f"Huấn luyện hoàn tất!")
        logger.info(f"Kích thước weights: {weights.shape}")
        logger.info(f"Bias (intercept): {bias:.6f}")

        # Đảm bảo số lượng weights khớp với số features
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]
        if len(weights) != len(feature_names):
            logger.warning(
                f"Số lượng weights ({len(weights)}) khác với số lượng features ({len(feature_names)}). "
                f"Sẽ điều chỉnh."
            )
            weights = weights[:len(feature_names)]

        # Tạo DataFrame lưu weights
        weights_df = pd.DataFrame({
            "feature": feature_names,
            "weight": weights
        })

        # Thêm bias vào cuối
        bias_entry = pd.DataFrame([{"feature": "bias", "weight": float(bias)}])
        weights_df = pd.concat([weights_df, bias_entry], ignore_index=True)

        # Lưu file
        weights_df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu model weights vào {output_path}")

        # In ra 10 trọng số quan trọng nhất
        weights_abs = weights_df.iloc[:-1].copy()
        weights_abs["abs_weight"] = weights_abs["weight"].abs()
        top_features = weights_abs.nlargest(10, "abs_weight")
        logger.info("Top 10 features quan trọng nhất:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['weight']:.6f}")

        return weights, bias


if __name__ == "__main__":
    train_model()
