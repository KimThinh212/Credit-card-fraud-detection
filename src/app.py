"""
FastAPI Backend - Credit Card Fraud Detection API
Chức năng:
  - CORS Middleware cho phép frontend gọi API
  - Pydantic validation cho input
  - Load model weights từ file vào bộ nhớ khi khởi động
  - Endpoint POST /predict: dự đoán gian lận giao dịch
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator


# ============================================================================
# Pydantic Models - Validation dữ liệu đầu vào
# ============================================================================

class TransactionInput(BaseModel):
    """
    Schema đầu vào cho API dự đoán.
    Gồm 28 thành phần PCA (V1-V28), Time và Amount.
    """
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    Time: float = Field(..., description="Thời gian (giây) kể từ giao dịch đầu tiên")
    Amount: float = Field(..., ge=0, description="Số tiền giao dịch (>= 0)")

    @validator("Amount")
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError("Amount phải >= 0")
        return v


class TransactionOutput(BaseModel):
    """
    Schema đầu ra của API dự đoán.
    """
    is_fraud: bool = Field(..., description="Giao dịch có phải gian lận không")
    fraud_probability: float = Field(..., description="Xác suất gian lận (0.0 - 1.0)")
    risk_level: str = Field(..., description="Mức độ rủi ro: Low / Medium / High")


# ============================================================================
# Global variables - Lưu model weights
# ============================================================================

model_weights: Optional[np.ndarray] = None
model_bias: float = 0.0
feature_order: List[str] = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
scaler_mean: Optional[pd.Series] = None
scaler_std: Optional[pd.Series] = None


# ============================================================================
# Helper functions
# ============================================================================

def compute_time_stats():
    """
    Tính mean và std của Time và Amount từ dữ liệu full (nếu có).
    Dùng để chuẩn hóa dữ liệu đầu vào giống như trong quá trình training.
    
    Nếu không tìm thấy file thống kê, dùng giá trị mặc định từ
    phân phối của Credit Card Fraud Detection dataset.
    """
    global scaler_mean, scaler_std

    stats_path = os.path.join(project_root, "data", "processed", "full_scaled.csv")
    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)
        scaler_mean = df[["Time", "Amount"]].mean()
        scaler_std = df[["Time", "Amount"]].std()
        logger.info("Đã tính mean/std từ dữ liệu full.")
    else:
        # Giá trị mặc định từ dataset (tham khảo từ EDA phổ biến)
        scaler_mean = pd.Series({"Time": 75000.0, "Amount": 88.0})
        scaler_std = pd.Series({"Time": 48000.0, "Amount": 250.0})
        logger.warning("Không tìm thấy full_scaled.csv, dùng giá trị mặc định cho scaling.")


def scale_input_time_amount(time: float, amount: float):
    """
    Chuẩn hóa Time và Amount giống như trong quá trình training.
    """
    global scaler_mean, scaler_std
    if scaler_mean is None or scaler_std is None:
        compute_time_stats()

    time_scaled = (time - scaler_mean["Time"]) / scaler_std["Time"]
    amount_scaled = (amount - scaler_mean["Amount"]) / scaler_std["Amount"]
    return time_scaled, amount_scaled


def sigmoid(z: float) -> float:
    """
    Hàm sigmoid để chuyển logit thành xác suất.
    """
    z = np.clip(z, -500, 500)  # Tránh overflow
    return 1.0 / (1.0 + np.exp(-z))


def predict(features: np.ndarray) -> tuple:
    """
    Dự đoán: nhân ma trận (dot product) + sigmoid
    
    Args:
        features: numpy array chứa 30 features (V1-V28, Time_scaled, Amount_scaled)
    
    Returns:
        (is_fraud, probability)
    """
    global model_weights, model_bias

    if model_weights is None:
        raise ValueError("Model weights chưa được load!")

    # Dot product: z = w^T * x + b
    logit = float(np.dot(model_weights, features)) + model_bias

    # Sigmoid -> xác suất
    probability = sigmoid(logit)

    # Phân loại
    is_fraud = probability >= 0.5

    return is_fraud, probability


def get_risk_level(probability: float) -> str:
    """
    Xác định mức độ rủi ro dựa trên xác suất.
      - Low:    < 30%
      - Medium: 30% - 70%
      - High:   > 70%
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def load_model_weights():
    """
    Load model weights từ file CSV vào bộ nhớ.
    File CSV có cấu trúc: feature, weight
    """
    global model_weights, model_bias

    weights_path = os.path.join(project_root, "data", "processed", "model_weights.csv")
    if not os.path.exists(weights_path):
        logger.error(f"Không tìm thấy file weights: {weights_path}")
        logger.error("Hãy chạy train_systemds.py trước khi khởi động API.")
        return False

    logger.info(f"Đang load model weights từ {weights_path}...")
    weights_df = pd.read_csv(weights_path)

    # Tách bias
    bias_row = weights_df[weights_df["feature"] == "bias"]
    if not bias_row.empty:
        model_bias = float(bias_row.iloc[0]["weight"])
        weights_df = weights_df[weights_df["feature"] != "bias"]
    else:
        model_bias = 0.0
        logger.warning("Không tìm thấy bias trong file weights, mặc định = 0.")

    # Tạo dictionary feature -> weight
    weight_dict = dict(zip(weights_df["feature"], weights_df["weight"]))

    # Sắp xếp đúng thứ tự features
    ordered_weights = []
    weight_summary = {}
    for feat in feature_order:
        w = weight_dict.get(feat, 0.0)
        ordered_weights.append(w)
        weight_summary[feat] = w

    model_weights = np.array(ordered_weights, dtype=np.float64)

    logger.info(f"Model weights loaded: {len(model_weights)} features + bias = {model_bias:.6f}")
    logger.info("Top 5 features có trọng số lớn nhất:")
    sorted_feats = sorted(weight_summary.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, w in sorted_feats[:5]:
        logger.info(f"  {feat}: {w:.6f}")

    compute_time_stats()
    return True


# ============================================================================
# FastAPI App với Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời của ứng dụng:
      - Startup: Load model weights
      - Shutdown: Cleanup (nếu cần)
    """
    logger.info("Khởi động API...")
    success = load_model_weights()
    if not success:
        logger.error("API khởi động nhưng KHÔNG có model weights!")
    yield
    logger.info("API đã tắt.")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API dự đoán gian lận giao dịch thẻ tín dụng sử dụng Logistic Regression (SystemDS)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware - Cho phép tất cả origin (dùng trong development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Endpoint gốc - kiểm tra API hoạt động
    """
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running",
        "model_loaded": model_weights is not None,
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """
    Endpoint kiểm tra sức khỏa - health check
    """
    return {
        "status": "healthy",
        "model_loaded": model_weights is not None,
        "num_features": len(model_weights) if model_weights is not None else 0
    }


@app.post("/predict", response_model=TransactionOutput)
async def predict_fraud(transaction: TransactionInput):
    """
    Dự đoán giao dịch có phải gian lận không.
    
    Quy trình:
      1. Chuẩn hóa Time và Amount
      2. Sắp xếp features theo đúng thứ tự (V1-V28, Time_scaled, Amount_scaled)
      3. Tính dot product với model weights
      4. Sigmoid -> xác suất
      5. Trả về kết quả
    """
    global model_weights, scaler_mean, scaler_std

    if model_weights is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa được load. Vui lòng chạy train_systemds.py trước."
        )

    try:
        # Chuẩn hóa Time và Amount
        time_scaled, amount_scaled = scale_input_time_amount(
            transaction.Time, transaction.Amount
        )

        # Xây dựng feature vector theo đúng thứ tự
        feature_values = [
            getattr(transaction, f"V{i}") for i in range(1, 29)
        ] + [time_scaled, amount_scaled]

        features = np.array(feature_values, dtype=np.float64)

        # Dự đoán
        is_fraud, probability = predict(features)
        risk_level = get_risk_level(probability)

        logger.info(
            f"Dự đoán: is_fraud={is_fraud}, "
            f"probability={probability:.4f}, "
            f"risk_level={risk_level}"
        )

        return TransactionOutput(
            is_fraud=is_fraud,
            fraud_probability=round(probability, 6),
            risk_level=risk_level
        )

    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


# ============================================================================
# Entry point (chạy trực tiếp)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Khởi động FastAPI server với uvicorn...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
