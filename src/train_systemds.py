"""
Huấn luyện mô hình Credit Card Fraud Detection sử dụng Apache SystemDS
Chức năng:
  - Đọc dữ liệu Train đã chuẩn hóa từ data/processed/
  - Huấn luyện mô hình Logistic Regression (l2svm) với SystemDS
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


def read_dataframe_with_spark(data_path, is_label=False):
    """
    Đọc CSV bằng pandas (fallback nếu SystemDS không đọc được trực tiếp).
    Trả về numpy array.
    
    SystemDS có thể đọc CSV trực tiếp qua ctx.read(), nhưng đôi khi
    cần định dạng đặc biệt. Hàm này đọc bằng pandas và chuyển thành
    numpy array, sau đó nạp vào SystemDS.
    """
    df = pd.read_csv(data_path)
    values = df.values.astype(np.float64)
    logger.info(f"Đã đọc {data_path}: shape = {values.shape}")
    return values


def train_model():
    """
    Huấn luyện mô hình sử dụng SystemDS l2svm (binary logistic regression)
    
    Quy trình:
      1. Đọc X_train, y_train từ CSV
      2. Nạp vào SystemDS dưới dạng matrix
      3. Huấn luyện bằng l2svm (L2-regularized SVM / Logistic Regression)
      4. Xuất weights và bias
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
    X_np = read_dataframe_with_spark(X_train_path, is_label=False)
    y_np = read_dataframe_with_spark(y_train_path, is_label=True)

    n_samples, n_features = X_np.shape
    logger.info(f"Số lượng mẫu: {n_samples}, Số lượng features: {n_features}")

    # Chuyển y thành vector 1D nếu cần
    if y_np.ndim > 1 and y_np.shape[1] == 1:
        y_np = y_np.flatten()

    # Khởi tạo SystemDS Context
    with SystemDSContext() as ctx:
        logger.info("Đã khởi tạo SystemDSContext.")

        # Nạp dữ liệu vào SystemDS
        X = ctx.from_numpy(X_np)
        y = ctx.from_numpy(y_np)

        logger.info("Đang huấn luyện mô hình với l2svm...")
        logger.info("  -> Thuật toán: l2svm (L2-regularized SVM / Logistic Regression)")
        logger.info("  -> maxi=200 (số vòng lặp tối đa)")
        logger.info("  -> tol=1e-7 (ngưỡng hội tụ)")
        logger.info("  -> reg=0.001 (hệ số regularization)")

        # Huấn luyện mô hình
        # l2svm trả về: (weights, bias)
        #   - weights: ma trận (n_features, 1)
        #   - bias: scalar
        weights_node, bias_node = ctx.l2svm(
            X, y,
            maxi=200,    # Số vòng lặp tối đa
            tol=1e-7,    # Ngưỡng hội tụ
            reg=0.001    # Hệ số regularization (L2)
        )

        # Lấy giá trị numpy
        weights = weights_node.compute()
        bias = bias_node.compute()

        logger.info(f"Huấn luyện hoàn tất!")
        logger.info(f"Kích thước weights: {weights.shape}")
        logger.info(f"Bias: {bias}")

        # Chuyển weights thành vector 1D nếu là ma trận cột
        if weights.ndim == 2 and weights.shape[1] == 1:
            weights = weights.flatten()

        # Tạo danh sách feature names
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]

        # Đảm bảo số lượng weights khớp với số features
        if len(weights) != len(feature_names):
            logger.warning(
                f"Số lượng weights ({len(weights)}) khác với số lượng features ({len(feature_names)}). "
                f"Sẽ điều chỉnh để phù hợp."
            )
            # Nếu weights nhiều hơn, lấy số lượng tương ứng
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
        weights_abs = weights_df.iloc[:-1].copy()  # Bỏ bias
        weights_abs["abs_weight"] = weights_abs["weight"].abs()
        top_features = weights_abs.nlargest(10, "abs_weight")
        logger.info("Top 10 features quan trọng nhất:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['weight']:.6f}")

        return weights, bias


if __name__ == "__main__":
    train_model()
