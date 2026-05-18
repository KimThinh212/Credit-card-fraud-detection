"""
FastAPI Backend - Credit Card Fraud Detection API (Enhanced)
Chức năng:
  - CORS Middleware
  - Pydantic validation cho input (TransactionInput, BatchTransactionInput)
  - Load model weights từ file vào bộ nhớ khi khởi động
  - Endpoint POST /predict: dự đoán gian lận giao dịch
  - Endpoint POST /predict/batch: dự đoán hàng loạt (tối đa 100)
  - Endpoint GET /stats: thống kê phiên làm việc
  - Endpoint POST /predict/explain: giải thích kết quả (feature contribution)
  - Endpoint GET /model/info: thông tin chi tiết model
  - Endpoint POST /reset/stats: reset thống kê phiên
  - Request timing middleware
  - Global exception handler
"""

import os
import sys
import logging
import time as time_module
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class TransactionInput(BaseModel):
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
    transaction_id: Optional[str] = Field(None, description="Mã giao dịch tùy chọn")
    merchant_category: Optional[str] = Field(None, description="Loại merchant tùy chọn")

    @field_validator("Amount")
    @classmethod
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError("Amount phải >= 0")
        return v

    @field_validator("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                      "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                      "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28")
    @classmethod
    def validate_v_features(cls, v):
        if v < -20 or v > 20:
            raise ValueError(f"Giá trị phải trong khoảng [-20, 20], nhận {v}")
        return v


class TransactionOutput(BaseModel):
    is_fraud: bool = Field(..., description="Giao dịch có phải gian lận không")
    fraud_probability: float = Field(..., description="Xác suất gian lận (0.0 - 1.0)")
    risk_level: str = Field(..., description="Mức độ rủi ro: Low / Medium / High")
    transaction_id: Optional[str] = Field(None, description="Mã giao dịch")
    timestamp: str = Field(..., description="Thời gian dự đoán (ISO format)")


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput] = Field(..., min_items=1, max_items=100)

    @field_validator("transactions")
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) < 1:
            raise ValueError("Phải có ít nhất 1 giao dịch")
        if len(v) > 100:
            raise ValueError("Tối đa 100 giao dịch mỗi lần gọi")
        return v


class BatchTransactionOutput(BaseModel):
    results: List[TransactionOutput]
    total: int
    fraud_count: int
    fraud_rate: float
    processing_time_ms: float


class ExplainOutput(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    top_features: List[dict]
    explanation_text: str


class ModelInfoOutput(BaseModel):
    num_features: int
    top_10_weights: List[dict]
    bias: float
    model_type: str
    framework: str


# ============================================================================
# Global variables
# ============================================================================

model_weights: Optional[np.ndarray] = None
model_bias: float = 0.0
feature_order: List[str] = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]
scaler_mean: Optional[pd.Series] = None
scaler_std: Optional[pd.Series] = None
startup_time: Optional[datetime] = None

session_stats = {
    "total_predictions": 0,
    "fraud_detected": 0,
    "safe_transactions": 0,
    "avg_probability": 0.0,
    "last_prediction_time": None,
    "high_risk_count": 0,
    "medium_risk_count": 0,
    "low_risk_count": 0,
}


# ============================================================================
# Helper functions
# ============================================================================

def compute_time_stats():
    global scaler_mean, scaler_std
    raw_path = os.path.join(project_root, "data", "raw", "creditcard.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, usecols=["Time", "Amount"])
        scaler_mean = df[["Time", "Amount"]].mean()
        scaler_std = df[["Time", "Amount"]].std()
        logger.info(f"Đã tính mean/std từ raw data: "
                     f"Time(mean={scaler_mean['Time']:.2f}, std={scaler_std['Time']:.2f}), "
                     f"Amount(mean={scaler_mean['Amount']:.2f}, std={scaler_std['Amount']:.2f})")
    else:
        scaler_mean = pd.Series({"Time": 75000.0, "Amount": 88.0})
        scaler_std = pd.Series({"Time": 48000.0, "Amount": 250.0})
        logger.warning("Không tìm thấy raw CSV, dùng giá trị mặc định cho scaling.")


def scale_input_time_amount(time_val: float, amount: float):
    global scaler_mean, scaler_std
    if scaler_mean is None or scaler_std is None:
        compute_time_stats()
    time_scaled = (time_val - scaler_mean["Time"]) / scaler_std["Time"]
    amount_scaled = (amount - scaler_mean["Amount"]) / scaler_std["Amount"]
    return time_scaled, amount_scaled


def sigmoid(z: float) -> float:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def predict(features: np.ndarray) -> tuple:
    global model_weights, model_bias
    if model_weights is None:
        raise ValueError("Model weights chưa được load!")
    logit = float(np.dot(model_weights, features)) + model_bias
    probability = sigmoid(logit)
    is_fraud = probability >= 0.5
    return is_fraud, probability


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def load_model_weights():
    global model_weights, model_bias
    weights_path = os.path.join(project_root, "data", "processed", "model_weights.csv")
    if not os.path.exists(weights_path):
        logger.error(f"Không tìm thấy file weights: {weights_path}")
        logger.error("Hãy chạy train_systemds.py trước khi khởi động API.")
        return False

    logger.info(f"Đang load model weights từ {weights_path}...")
    weights_df = pd.read_csv(weights_path)

    bias_row = weights_df[weights_df["feature"] == "bias"]
    if not bias_row.empty:
        model_bias = float(bias_row.iloc[0]["weight"])
        weights_df = weights_df[weights_df["feature"] != "bias"]
    else:
        model_bias = 0.0
        logger.warning("Không tìm thấy bias, mặc định = 0.")

    weight_dict = dict(zip(weights_df["feature"], weights_df["weight"]))
    ordered_weights = []
    weight_summary = {}
    for feat in feature_order:
        w = weight_dict.get(feat, 0.0)
        ordered_weights.append(w)
        weight_summary[feat] = w

    model_weights = np.array(ordered_weights, dtype=np.float64)
    logger.info(f"Model weights loaded: {len(model_weights)} features + bias = {model_bias:.6f}")

    compute_time_stats()
    return True


def update_stats(is_fraud: bool, probability: float, risk_level: str):
    session_stats["total_predictions"] += 1
    if is_fraud:
        session_stats["fraud_detected"] += 1
    else:
        session_stats["safe_transactions"] += 1
    total = session_stats["total_predictions"]
    old_avg = session_stats["avg_probability"]
    session_stats["avg_probability"] = (old_avg * (total - 1) + probability) / total
    session_stats["last_prediction_time"] = datetime.now(timezone.utc).isoformat()
    if risk_level == "High":
        session_stats["high_risk_count"] += 1
    elif risk_level == "Medium":
        session_stats["medium_risk_count"] += 1
    else:
        session_stats["low_risk_count"] += 1


def build_transaction_output(is_fraud: bool, probability: float, risk_level: str,
                             transaction_id: Optional[str] = None) -> dict:
    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(probability, 6),
        "risk_level": risk_level,
        "transaction_id": transaction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_time
    startup_time = datetime.now(timezone.utc)
    logger.info("Khởi động API...")
    success = load_model_weights()
    if not success:
        logger.error("API khởi động nhưng KHÔNG có model weights!")
    yield
    logger.info("API đã tắt.")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API dự đoán gian lận giao dịch thẻ tín dụng sử dụng Logistic Regression (SystemDS)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Middleware: Request timing
# ============================================================================

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time_module.time()
    response = await call_next(request)
    elapsed = time_module.time() - start
    response.headers["X-Process-Time-Ms"] = str(round(elapsed * 1000, 2))
    return response


# ============================================================================
# Global exception handler
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running",
        "model_loaded": model_weights is not None,
        "version": "2.0.0",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_weights is not None,
        "num_features": len(model_weights) if model_weights is not None else 0,
    }


@app.post("/predict", response_model=TransactionOutput)
async def predict_fraud(transaction: TransactionInput):
    global model_weights
    if model_weights is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")

    try:
        time_scaled, amount_scaled = scale_input_time_amount(transaction.Time, transaction.Amount)
        feature_values = [getattr(transaction, f"V{i}") for i in range(1, 29)] + [time_scaled, amount_scaled]
        features = np.array(feature_values, dtype=np.float64)
        is_fraud, probability = predict(features)
        risk_level = get_risk_level(probability)
        update_stats(is_fraud, probability, risk_level)

        logger.info(f"Dự đoán: is_fraud={is_fraud}, probability={probability:.4f}, risk_level={risk_level}")

        return build_transaction_output(is_fraud, probability, risk_level, transaction.transaction_id)

    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


@app.post("/predict/batch", response_model=BatchTransactionOutput)
async def predict_batch(batch: BatchTransactionInput):
    global model_weights
    if model_weights is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")

    start = time_module.time()
    results = []

    for tx in batch.transactions:
        try:
            time_scaled, amount_scaled = scale_input_time_amount(tx.Time, tx.Amount)
            feature_values = [getattr(tx, f"V{i}") for i in range(1, 29)] + [time_scaled, amount_scaled]
            features = np.array(feature_values, dtype=np.float64)
            is_fraud, probability = predict(features)
            risk_level = get_risk_level(probability)
            update_stats(is_fraud, probability, risk_level)
            results.append(build_transaction_output(is_fraud, probability, risk_level, tx.transaction_id))
        except Exception as e:
            logger.error(f"Lỗi xử lý giao dịch trong batch: {e}")
            results.append(build_transaction_output(False, 0.0, "Error", tx.transaction_id))

    elapsed_ms = (time_module.time() - start) * 1000
    fraud_count = sum(1 for r in results if r["is_fraud"])
    total = len(results)

    return BatchTransactionOutput(
        results=results,
        total=total,
        fraud_count=fraud_count,
        fraud_rate=fraud_count / total if total > 0 else 0.0,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.get("/stats")
async def get_stats():
    global startup_time
    uptime_seconds = 0.0
    if startup_time:
        uptime_seconds = (datetime.now(timezone.utc) - startup_time).total_seconds()
    return {
        **session_stats,
        "uptime_seconds": round(uptime_seconds, 1),
        "startup_time": startup_time.isoformat() if startup_time else None,
    }


@app.post("/predict/explain", response_model=ExplainOutput)
async def predict_explain(transaction: TransactionInput):
    global model_weights, model_bias
    if model_weights is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")

    try:
        time_scaled, amount_scaled = scale_input_time_amount(transaction.Time, transaction.Amount)
        feature_values = [getattr(transaction, f"V{i}") for i in range(1, 29)] + [time_scaled, amount_scaled]
        features = np.array(feature_values, dtype=np.float64)
        is_fraud, probability = predict(features)
        risk_level = get_risk_level(probability)

        contributions = []
        for i, feat_name in enumerate(feature_order):
            if i < len(feature_values) and i < len(model_weights):
                contribution = float(model_weights[i] * feature_values[i])
                contributions.append({
                    "feature": feat_name,
                    "weight": round(float(model_weights[i]), 6),
                    "value": round(float(feature_values[i]), 6),
                    "contribution": round(contribution, 6),
                })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_features = contributions[:5]

        fraud_contribs = [c for c in contributions if c["contribution"] > 0]
        fraud_contribs.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        if is_fraud and fraud_contribs:
            top_names = [c["feature"] for c in fraud_contribs[:3]]
            top_vals = [f"{c['feature']} ({c['contribution']:+.4f})" for c in fraud_contribs[:3]]
            explanation_text = (
                f"Giao dịch bị đánh dấu {'rủi ro cao' if risk_level == 'High' else 'có nghi vấn'} "
                f"(xác suất {probability*100:.1f}%). "
                f"Yếu tố đóng góp nhiều nhất: {', '.join(top_vals)}."
            )
        elif not is_fraud:
            explanation_text = (
                f"Giao dịch an toàn (xác suất {probability*100:.1f}%). "
                f"Các đặc trưng có xu hướng ủng hộ giao dịch hợp lệ."
            )
        else:
            explanation_text = (
                f"Kết quả phân tích với xác suất {probability*100:.1f}%. "
                f"Mức rủi ro: {risk_level}."
            )

        return ExplainOutput(
            is_fraud=is_fraud,
            fraud_probability=round(probability, 6),
            risk_level=risk_level,
            top_features=top_features,
            explanation_text=explanation_text,
        )

    except Exception as e:
        logger.error(f"Lỗi khi explain: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


@app.get("/model/info", response_model=ModelInfoOutput)
async def model_info():
    global model_weights, model_bias
    if model_weights is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")

    weight_summary = []
    for i, feat in enumerate(feature_order):
        if i < len(model_weights):
            weight_summary.append({
                "feature": feat,
                "abs_weight": round(float(abs(model_weights[i])), 6),
                "weight": round(float(model_weights[i]), 6),
            })

    weight_summary.sort(key=lambda x: x["abs_weight"], reverse=True)
    top_10 = weight_summary[:10]

    return ModelInfoOutput(
        num_features=len(feature_order),
        top_10_weights=top_10,
        bias=round(float(model_bias), 6),
        model_type="Logistic Regression",
        framework="SystemDS",
    )


@app.post("/reset/stats")
async def reset_stats():
    global session_stats
    session_stats = {
        "total_predictions": 0,
        "fraud_detected": 0,
        "safe_transactions": 0,
        "avg_probability": 0.0,
        "last_prediction_time": None,
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "low_risk_count": 0,
    }
    return {
        "message": "Stats đã được reset",
        "reset_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("Khởi động FastAPI server với uvicorn...")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
