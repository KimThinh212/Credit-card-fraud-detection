"""
Streamlit Dashboard - Credit Card Fraud Detection (Enhanced)
Chức năng:
  - Dashboard tổng quan với biểu đồ Plotly (dark theme)
  - Form dự đoán realtime với transaction_id
  - Batch analysis với CSV upload
  - Explain AI với feature contribution
  - Lịch sử giao dịch
  - Random transaction generator (Normal / Suspicious)
  - Session stats từ API
"""

import os
import sys
import json
import random
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# Cấu hình trang
# ============================================================================

st.set_page_config(
    page_title="FraudShield – Credit Card Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Constants
# ============================================================================

API_URL = os.environ.get("API_URL", "http://localhost:8000")
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)]

V_FRAUD_INDICES = {3, 4, 9, 10, 11, 12, 14, 16, 17}


# ============================================================================
# Logo
# ============================================================================

def get_logo_base64() -> Optional[str]:
    logo_paths = [
        Path(project_root) / "logo_credit-card.jpg",
        Path(__file__).parent.parent.parent / "logo_credit-card.jpg",
        Path(__file__).parent.parent / "logo_credit-card.jpg",
        Path("logo_credit-card.jpg"),
        Path("E:/BigData/credit-card-fraud-detection/logo_credit-card.jpg"),
    ]
    seen = set()
    for p in logo_paths:
        resolved = p.resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        try:
            if p.exists():
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode()
        except Exception:
            continue
    return None


logo_b64 = get_logo_base64()


# ============================================================================
# Custom CSS – Dark Professional Theme
# ============================================================================

st.markdown("""
<style>
    :root {
        --bg-primary: #0A0E1A;
        --bg-card: #111827;
        --bg-card-hover: #1a2233;
        --accent-blue: #3B82F6;
        --accent-green: #10B981;
        --accent-red: #EF4444;
        --accent-orange: #F59E0B;
        --accent-purple: #8B5CF6;
        --text-primary: #F9FAFB;
        --text-secondary: #9CA3AF;
        --border: #1F2937;
        --gradient-fraud: linear-gradient(135deg, #EF4444, #7C3AED);
        --gradient-safe: linear-gradient(135deg, #10B981, #3B82F6);
        --gradient-medium: linear-gradient(135deg, #F59E0B, #EF4444);
    }

    .stApp { background: var(--bg-primary) !important; }

    [data-testid="stSidebar"] {
        background: #0D1117 !important;
        border-right: 1px solid var(--border) !important;
    }

    .glass-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        margin-bottom: 1rem;
    }

    .metric-glass {
        background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.1));
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-glass:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: var(--text-primary); }
    .metric-label { font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px; }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
        50% { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
    }
    .alert-fraud {
        background: var(--gradient-fraud);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        animation: pulse-red 2s infinite;
    }
    .alert-safe {
        background: var(--gradient-safe);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .alert-medium-box {
        background: var(--gradient-medium);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .risk-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 50px;
        height: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(17,24,39,0.8);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        color: white !important;
    }

    .stNumberInput input, .stTextInput input {
        background: rgba(17,24,39,0.9) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background: rgba(17,24,39,0.9) !important;
        border: 1px solid var(--border) !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
        color: white !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59,130,246,0.4) !important;
    }

    .page-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .page-sub {
        color: var(--text-secondary);
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-high   { background: rgba(239,68,68,0.2);  color: #EF4444; }
    .badge-medium { background: rgba(245,158,11,0.2); color: #F59E0B; }
    .badge-low    { background: rgba(16,185,129,0.2); color: #10B981; }

    div[data-testid="stMetric"] {
        background: rgba(17,24,39,0.6);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
    }
    div[data-testid="stMetric"] div {
        color: var(--text-primary) !important;
    }

    .stDataFrame { border-radius: 8px; overflow: hidden; }

    .stFileUploader div[data-testid="stFileUploadDropzone"] {
        background: rgba(17,24,39,0.6) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 12px !important;
    }

    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper functions
# ============================================================================

def get_dashboard_data():
    np.random.seed(42)
    n = 1000
    amounts = np.clip(np.random.exponential(scale=100, size=n), 1, 2000)
    times = np.random.uniform(0, 172792, size=n)
    fraud_ratio = 0.017
    classes = np.random.choice([0, 1], size=n, p=[1-fraud_ratio, fraud_ratio])
    v_data = {}
    for i in range(1, 29):
        v_mean = 0.5 if i in V_FRAUD_INDICES else 0.0
        v_data[f"V{i}"] = np.random.normal(loc=np.where(classes == 1, v_mean, 0.0), scale=1.0, size=n)
    df = pd.DataFrame({"Time": times, "Amount": amounts, "Class": classes, **v_data})
    df.loc[df["Class"] == 1, "Amount"] *= 2.5
    return df


def generate_random_transaction(is_fraud_sim: Optional[bool] = None) -> dict:
    if is_fraud_sim is None:
        is_fraud_sim = random.random() < 0.1
    tx = {}
    for i in range(1, 29):
        if is_fraud_sim and i in V_FRAUD_INDICES:
            tx[f"V{i}"] = round(random.gauss(-1.5, 1.2), 6)
        else:
            tx[f"V{i}"] = round(random.gauss(0.0, 1.0), 6)
    tx["Time"] = round(random.uniform(0, 172792), 2)
    if is_fraud_sim:
        tx["Amount"] = round(random.uniform(500, 5000), 2)
    else:
        tx["Amount"] = round(random.uniform(1, 300), 2)
    return tx


def generate_normal_transaction() -> dict:
    tx = {}
    for i in range(1, 29):
        tx[f"V{i}"] = round(random.gauss(0.0, 1.0), 6)
    tx["Time"] = round(random.uniform(0, 172792), 2)
    tx["Amount"] = round(random.uniform(5, 300), 2)
    return tx


def generate_suspicious_transaction() -> dict:
    tx = {}
    for i in range(1, 29):
        if i in {3, 4, 10, 12, 14}:
            tx[f"V{i}"] = round(random.gauss(-2.5, 1.0), 6)
        elif i in {9, 11, 16, 17}:
            tx[f"V{i}"] = round(random.gauss(-1.8, 1.2), 6)
        else:
            tx[f"V{i}"] = round(random.gauss(0.0, 0.8), 6)
    tx["Time"] = round(random.uniform(0, 50000), 2)
    tx["Amount"] = round(random.uniform(500, 5000), 2)
    return tx


def generate_transaction_id() -> str:
    import uuid
    return f"TX-{uuid.uuid4().hex[:8].upper()}"


def call_api(endpoint: str, method: str = "GET", json_data: dict = None,
             timeout: int = 10) -> Optional[dict]:
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=json_data, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API lỗi ({resp.status_code}): {resp.text[:200]}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Không thể kết nối đến API tại {API_URL}")
        return None
    except Exception as e:
        st.error(f"❌ Lỗi kết nối: {e}")
        return None


def build_gauge_chart(prob_pct: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_pct,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"color": color, "size": 40}, "suffix": "%"},
        delta={"reference": 50, "increasing": {"color": "#EF4444"},
               "decreasing": {"color": "#10B981"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9CA3AF"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "rgba(16,185,129,0.3)"},
                {"range": [30, 70], "color": "rgba(245,158,11,0.3)"},
                {"range": [70, 100], "color": "rgba(239,68,68,0.3)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#9CA3AF"},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def apply_dark_layout(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#9CA3AF"},
        margin=dict(l=20, r=20, t=30, b=20),
        height=height,
    )
    return fig


def format_risk_badge(risk_level: str) -> str:
    level = risk_level.lower() if risk_level else "low"
    if level == "high":
        return f'<span class="badge badge-high">HIGH</span>'
    elif level == "medium":
        return f'<span class="badge badge-medium">MEDIUM</span>'
    else:
        return f'<span class="badge badge-low">LOW</span>'


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    with st.sidebar:
        if logo_b64:
            st.markdown(
                f'<div style="text-align:center; margin-bottom:1rem;">'
                f'<img src="data:image/jpeg;base64,{logo_b64}" '
                f'style="width:120px; border-radius:12px; '
                f'box-shadow:0 4px 16px rgba(59,130,246,0.4);" /></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:center; font-size:3rem; margin-bottom:0.5rem;">🛡️</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="page-header" style="font-size:1.4rem; text-align:center;">FraudShield</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="page-sub" style="text-align:center;">AI Fraud Detection</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("#### 🔌 Trạng thái hệ thống")
        health_data = call_api("/health", timeout=3)
        if health_data:
            st.success("🟢 API Online")
            st.caption(f"Features: {health_data.get('num_features', 30)}")
        else:
            st.error("🔴 API Offline")
            st.caption(f"Kiểm tra `{API_URL}`")

        st.markdown("---")

        st.markdown("#### 📊 Thống kê phiên")
        stats = call_api("/stats", timeout=3)
        if stats:
            c1, c2 = st.columns(2)
            c1.metric("Tổng GD", stats.get("total_predictions", 0))
            c2.metric("Gian lận", stats.get("fraud_detected", 0))
            avg_prob = stats.get("avg_probability", 0.0)
            st.progress(avg_prob, text=f"Xác suất TB: {avg_prob*100:.1f}%")
            uptime = stats.get("uptime_seconds", 0)
            if uptime:
                mins = int(uptime // 60)
                secs = int(uptime % 60)
                st.caption(f"Uptime: {mins}m {secs}s")
        else:
            st.caption("Chưa có dữ liệu phiên")

        if st.button("🔄 Reset Stats", use_container_width=True):
            resp = call_api("/reset/stats", method="POST", timeout=3)
            if resp:
                st.success("✅ Đã reset!")
                st.rerun()

        st.markdown("---")

        st.markdown("#### 🤖 Thông tin Model")
        info = call_api("/model/info", timeout=3)
        if info:
            st.info(
                f"**Loại:** {info.get('model_type', 'Logistic Regression')}  \n"
                f"**Framework:** {info.get('framework', 'SystemDS')}  \n"
                f"**Features:** {info.get('num_features', 30)}  \n"
                f"**Bias:** {info.get('bias', 0.0):.4f}"
            )
        else:
            st.info("**Model:** Logistic Regression\n**Features:** 30")

        st.markdown("---")
        st.caption("FraudShield v2.0 | © 2026 Credit Card Protection Team")


# ============================================================================
# TAB 1 – Dashboard
# ============================================================================

def render_dashboard():
    st.markdown('<div class="page-header">📊 Dashboard Tổng quan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Phân tích giao dịch & Phát hiện gian lận thời gian thực</div>',
        unsafe_allow_html=True,
    )

    stats = call_api("/stats", timeout=3)
    total_preds = stats.get("total_predictions", 0) if stats else 0
    fraud_detected = stats.get("fraud_detected", 0) if stats else 0
    safe_tx = stats.get("safe_transactions", 0) if stats else 0
    avg_prob = stats.get("avg_probability", 0.0) if stats else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f'<div class="metric-glass"><div class="metric-value">{total_preds}</div>'
        f'<div class="metric-label">Tổng GD phiên này</div></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="metric-glass"><div class="metric-value">{fraud_detected}</div>'
        f'<div class="metric-label">Gian lận phát hiện</div></div>',
        unsafe_allow_html=True,
    )
    fraud_rate = (fraud_detected / total_preds * 100) if total_preds > 0 else 0.0
    col3.markdown(
        f'<div class="metric-glass"><div class="metric-value">{fraud_rate:.1f}%</div>'
        f'<div class="metric-label">Tỷ lệ gian lận</div></div>',
        unsafe_allow_html=True,
    )
    col4.markdown(
        f'<div class="metric-glass"><div class="metric-value">{avg_prob*100:.1f}%</div>'
        f'<div class="metric-label">Xác suất TB</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    df = get_dashboard_data()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Phân bố Amount theo Class")
        fig = px.histogram(
            df, x="Amount", color="Class",
            color_discrete_map={0: "#10B981", 1: "#EF4444"},
            nbins=50, opacity=0.7,
            labels={"Amount": "Số tiền (USD)"},
        )
        fig = apply_dark_layout(fig, height=320)
        fig.update_layout(xaxis=dict(range=[0, 500]))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Tỷ lệ An toàn vs Gian lận")
        legit = int(df["Class"].value_counts().get(0, 0))
        fraud = int(df["Class"].value_counts().get(1, 0))
        fig = go.Figure(go.Pie(
            labels=["An toàn", "Gian lận"],
            values=[legit, fraud],
            marker_colors=["#10B981", "#EF4444"],
            hole=0.45,
            textinfo="label+percent",
        ))
        fig = apply_dark_layout(fig, height=320)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Giao dịch theo thời gian (mô phỏng)")
        timeline = df.copy()
        timeline = timeline.sort_values("Time")
        timeline["tx_index"] = range(len(timeline))
        fig = px.scatter(
            timeline, x="tx_index", y="Amount", color="Class",
            color_discrete_map={0: "rgba(16,185,129,0.3)", 1: "#EF4444"},
            opacity=0.6, size_max=8,
            labels={"tx_index": "Giao dịch (theo thời gian)", "Amount": "Số tiền (USD)"},
        )
        fig = apply_dark_layout(fig, height=280)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔥 Correlation Matrix (V1-V10)")
        corr = df[[f"V{i}" for i in range(1, 11)]].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu",
            aspect="auto", zmin=-1, zmax=1,
        )
        fig = apply_dark_layout(fig, height=280)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# TAB 2 – Kiểm tra Giao dịch
# ============================================================================

def render_prediction_form():
    st.markdown('<div class="page-header">🔍 Kiểm tra Giao dịch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Nhập thông tin giao dịch để kiểm tra gian lận</div>',
        unsafe_allow_html=True,
    )

    rand_tx = st.session_state.get("random_transaction")

    # Gán giá trị từ rand_tx vào session_state để widgets cập nhật
    if rand_tx:
        st.session_state["pred_amount"] = rand_tx.get("Amount", 100.0)
        st.session_state["pred_time"] = rand_tx.get("Time", 75000.0)
        for i in range(1, 29):
            st.session_state[f"pred_v{i}"] = rand_tx.get(f"V{i}", 0.0)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 💰 Thông tin giao dịch")

        r1, r2 = st.columns(2)
        with r1:
            amount = st.number_input("💰 Amount ($)", min_value=0.0, max_value=50000.0,
                                     value=100.0, step=1.0, format="%.2f", key="pred_amount")
        with r2:
            time_val = st.number_input("⏱ Time (giây)", min_value=0.0, max_value=200000.0,
                                       value=75000.0, step=1.0, format="%.2f", key="pred_time")

        tx_id = st.text_input("🔖 Mã giao dịch (tùy chọn)", placeholder="TX-...",
                              value=f"TX-{datetime.now().strftime('%H%M%S')}")

        with st.expander("🔬 PCA Components (V1-V28)", expanded=False):
            v_values = {}
            for row in range(7):
                cols = st.columns(4)
                for c in range(4):
                    idx = row * 4 + c + 1
                    if idx <= 28:
                        with cols[c]:
                            v_values[f"V{idx}"] = st.number_input(
                                f"V{idx}", value=0.0,
                                min_value=-10.0, max_value=10.0,
                                step=0.01, format="%.4f", key=f"pred_v{idx}",
                            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_left:
        b1, b2, b3 = st.columns(3)
        with b1:
            rand_normal = st.button("🎲 Random (Normal)", use_container_width=True)
        with b2:
            rand_sus = st.button("⚠️ Random (Suspicious)", use_container_width=True)
        with b3:
            predict_btn = st.button("🚀 Phân tích", type="primary", use_container_width=True)

    if rand_normal:
        st.session_state["random_transaction"] = generate_normal_transaction()
        st.session_state.pop("predict_result", None)
        st.rerun()

    if rand_sus:
        st.session_state["random_transaction"] = generate_suspicious_transaction()
        st.session_state.pop("predict_result", None)
        st.rerun()

    if rand_tx:
        st.info(f"🎲 Giao dịch ngẫu nhiên: Amount=${rand_tx.get('Amount', 0):.2f}, Time={rand_tx.get('Time', 0):.0f}s")

    with col_right:
        stored = st.session_state.get("predict_result")

        if predict_btn:
            transaction = {
                **{f"V{i}": v_values.get(f"V{i}", 0.0) for i in range(1, 29)},
                "Time": time_val,
                "Amount": amount,
                "transaction_id": tx_id,
            }

            with st.spinner("🔄 Đang phân tích..."):
                result = call_api("/predict", method="POST", json_data=transaction)

            if result:
                st.session_state["predict_result"] = result
                st.session_state["predict_tx_id"] = tx_id
                st.session_state["predict_amount"] = amount

                history = st.session_state.get("history", [])
                history.append({
                    "id": tx_id or f"TX-{len(history)+1:04d}",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "amount": amount,
                    "is_fraud": result["is_fraud"],
                    "probability": result["fraud_probability"],
                    "risk_level": result["risk_level"],
                })
                st.session_state["history"] = history

                st.rerun()

        stored = st.session_state.get("predict_result")
        if stored:
            is_fraud = stored["is_fraud"]
            probability = stored["fraud_probability"]
            risk_level = stored["risk_level"]
            prob_pct = probability * 100

            gauge_color = "#EF4444" if risk_level == "High" else "#F59E0B" if risk_level == "Medium" else "#10B981"
            fig = build_gauge_chart(prob_pct, gauge_color)
            st.plotly_chart(fig, use_container_width=True)

            if risk_level == "High":
                st.markdown(
                    f'<div class="alert-fraud">🚨 GIAN LẬN – {prob_pct:.2f}%</div>',
                    unsafe_allow_html=True,
                )
            elif risk_level == "Medium":
                st.markdown(
                    f'<div class="alert-medium-box">⚠️ NGHI VẤN – {prob_pct:.2f}%</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="alert-safe">✅ AN TOÀN – {prob_pct:.2f}%</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div style="text-align:center; margin-top:0.5rem;">{format_risk_badge(risk_level)}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("🔎 Chi tiết"):
                tx_display = st.session_state.get("predict_tx_id", "N/A")
                amt_display = st.session_state.get("predict_amount", 0.0)
                st.json({
                    "is_fraud": is_fraud,
                    "fraud_probability": probability,
                    "risk_level": risk_level,
                    "transaction_id": tx_display,
                    "input": {"Amount": amt_display},
                })

        else:
            st.markdown(
                '<div style="text-align:center; padding:3rem; color:#9CA3AF;">'
                '👈 Nhập thông tin và bấm "Phân tích"</div>',
                unsafe_allow_html=True,
            )


# ============================================================================
# TAB 3 – Batch Analysis
# ============================================================================

def render_batch_tab():
    st.markdown('<div class="page-header">📦 Phân tích Hàng loạt</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Tải lên CSV hoặc dùng dữ liệu demo để phân tích nhiều giao dịch cùng lúc</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Tải lên file CSV (tối đa 100 giao dịch)",
        type=["csv"],
        help="File CSV cần có các cột: V1-V28, Time, Amount",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        use_demo = st.button("📋 Dùng dữ liệu Demo", use_container_width=True)

    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Lỗi đọc CSV: {e}")

    if use_demo:
        df = pd.DataFrame([generate_normal_transaction() for _ in range(20)])

    if df is not None:
        required_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"Thiếu cột: {', '.join(missing)}")
        else:
            st.success(f"✅ Đã tải {len(df)} giao dịch")
            st.dataframe(df.head(5), use_container_width=True)

            if st.button("🚀 Phân tích Batch", type="primary", use_container_width=True):
                with st.spinner("Đang xử lý..."):
                    transactions = df[required_cols].head(100).to_dict(orient="records")
                    response = call_api("/predict/batch", method="POST",
                                        json_data={"transactions": transactions}, timeout=30)

                if response:
                    results_list = response["results"]

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Tổng", response["total"])
                    c2.metric("Gian lận", response["fraud_count"],
                              delta=f"{response['fraud_rate']*100:.1f}%", delta_color="inverse")
                    c3.metric("An toàn", response["total"] - response["fraud_count"])
                    c4.metric("Thời gian", f"{response['processing_time_ms']:.1f}ms")

                    result_df = df.head(len(results_list)).copy()
                    result_df["is_fraud"] = [r["is_fraud"] for r in results_list]
                    result_df["probability"] = [r["fraud_probability"] for r in results_list]
                    result_df["risk_level"] = [r["risk_level"] for r in results_list]

                    def style_row(row):
                        if row["is_fraud"]:
                            return ["background-color: rgba(239,68,68,0.1)"] * len(row)
                        return ["background-color: rgba(16,185,129,0.1)"] * len(row)

                    display_cols = ["Amount", "Time", "is_fraud", "probability", "risk_level"]
                    st.dataframe(
                        result_df[display_cols].style.apply(style_row, axis=1),
                        use_container_width=True,
                    )

                    fig = go.Figure(go.Pie(
                        labels=["An toàn", "Gian lận"],
                        values=[response["total"] - response["fraud_count"], response["fraud_count"]],
                        marker_colors=["#10B981", "#EF4444"],
                        hole=0.45,
                    ))
                    fig = apply_dark_layout(fig, height=300)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    csv_out = result_df[display_cols].to_csv(index=False)
                    st.download_button(
                        "⬇️ Tải kết quả CSV",
                        csv_out,
                        "batch_results.csv",
                        "text/csv",
                        use_container_width=True,
                    )


# ============================================================================
# TAB 4 – Explain AI
# ============================================================================

def render_explain_tab():
    st.markdown('<div class="page-header">🧠 Giải thích AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Xem mô hình quyết định dựa trên đặc trưng nào</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 🏆 Top 10 Features Quan trọng nhất")
    info = call_api("/model/info", timeout=5)
    if info and info.get("top_10_weights"):
        top_list = info["top_10_weights"]
        features = [t["feature"] for t in top_list]
        weights = [t["abs_weight"] for t in top_list]
        colors = ["#EF4444" if w > 0.5 else "#F59E0B" if w > 0.2 else "#10B981" for w in weights]

        fig = go.Figure(go.Bar(
            x=weights, y=features, orientation="h",
            marker_color=colors,
            text=[f"{w:.4f}" for w in weights],
            textposition="outside",
        ))
        fig = apply_dark_layout(fig, height=350)
        fig.update_layout(xaxis_title="Trọng số tuyệt đối", margin=dict(l=20, r=80, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Không thể tải thông tin model. Đảm bảo API đang chạy.")

    st.markdown("---")

    st.markdown("### 🔎 Giải thích Giao dịch Cụ thể")
    history = st.session_state.get("history", [])

    col1, col2 = st.columns([2, 1])
    with col1:
        exp_amount = st.number_input("Amount", value=100.0, min_value=0.0, key="exp_amount")
        exp_time = st.number_input("Time", value=75000.0, key="exp_time")
    with col2:
        if st.button("🎲 Random", key="exp_random"):
            rand = generate_random_transaction()
            st.session_state["explain_data"] = rand
            st.rerun()

    with st.expander("🔬 PCA Components", expanded=False):
        exp_v = {}
        for row in range(7):
            cols = st.columns(4)
            for c in range(4):
                idx = row * 4 + c + 1
                if idx <= 28:
                    with cols[c]:
                        default = st.session_state.get("explain_data", {}).get(f"V{idx}", 0.0)
                        exp_v[f"V{idx}"] = st.number_input(
                            f"V{idx}", value=float(default),
                            min_value=-10.0, max_value=10.0,
                            key=f"exp_v{idx}", format="%.4f",
                        )

    if st.button("🧠 Giải thích Giao dịch này", type="primary", use_container_width=True):
        payload = {**exp_v, "Time": exp_time, "Amount": exp_amount}
        with st.spinner("Đang phân tích..."):
            data = call_api("/predict/explain", method="POST", json_data=payload, timeout=10)

        if data:
            top_feats = data.get("top_features", [])

            if data["is_fraud"]:
                st.markdown(
                    f'<div class="alert-fraud">🚨 GIAN LẬN – {data["fraud_probability"]*100:.2f}%</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="alert-safe">✅ AN TOÀN – {data["fraud_probability"]*100:.2f}%</div>',
                    unsafe_allow_html=True,
                )

            st.info(f"📝 {data.get('explanation_text', '')}")

            if top_feats:
                feat_names = [f["feature"] for f in top_feats]
                contributions = [f["contribution"] for f in top_feats]
                contrib_colors = ["#EF4444" if c < 0 else "#10B981" for c in contributions]

                fig = go.Figure(go.Bar(
                    x=feat_names, y=contributions,
                    marker_color=contrib_colors,
                    text=[f"{c:+.4f}" for c in contributions],
                    textposition="outside",
                ))
                fig = apply_dark_layout(fig, height=350)
                fig.update_layout(
                    title="Đóng góp của từng đặc trưng",
                    yaxis_title="Contribution (weight × value)",
                    shapes=[{
                        "type": "line", "x0": -0.5, "x1": len(feat_names) - 0.5,
                        "y0": 0, "y1": 0,
                        "line": {"color": "white", "width": 1, "dash": "dash"},
                    }],
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    pd.DataFrame(top_feats)[["feature", "weight", "value", "contribution"]],
                    use_container_width=True,
                )


# ============================================================================
# TAB 5 – Lịch sử
# ============================================================================

def render_history_tab():
    st.markdown('<div class="page-header">📜 Lịch sử Giao dịch</div>', unsafe_allow_html=True)
    history = st.session_state.get("history", [])

    if not history:
        st.info("Chưa có giao dịch nào. Hãy kiểm tra một giao dịch ở tab 'Kiểm tra GD'.")
        return

    total = len(history)
    fraud_count = sum(1 for h in history if h["is_fraud"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng giao dịch", total)
    col2.metric("Phát hiện gian lận", fraud_count)
    col3.metric("Tỷ lệ gian lận", f"{fraud_count/total*100:.1f}%" if total > 0 else "0%")

    if len(history) > 1:
        hist_df = pd.DataFrame(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(hist_df))),
            y=hist_df["probability"],
            mode="lines+markers",
            marker=dict(
                color=["#EF4444" if f else "#10B981" for f in hist_df["is_fraud"]],
                size=10, line=dict(color="white", width=1),
            ),
            line=dict(color="rgba(59,130,246,0.4)", width=2),
            name="Fraud Probability",
        ))
        fig.add_hline(
            y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.3)",
            annotation_text="Ngưỡng 50%",
        )
        fig = apply_dark_layout(fig, height=280)
        fig.update_layout(
            title="Xác suất Gian lận theo Thời gian",
            yaxis=dict(range=[0, 1], title="Xác suất"),
            xaxis_title="Giao dịch #",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Chi tiết")
    for item in reversed(history[-50:]):
        color = "#EF4444" if item["is_fraud"] else "#10B981"
        icon = "🚨" if item["is_fraud"] else "✅"
        risk_badge = format_risk_badge(item["risk_level"])
        st.markdown(
            f'<div style="display:flex; align-items:center; padding:0.6rem 1rem; '
            f'border-left: 3px solid {color}; background: rgba(255,255,255,0.03); '
            f'border-radius: 0 8px 8px 0; margin-bottom: 6px;">'
            f'<span style="font-size:1.2rem; margin-right:0.7rem;">{icon}</span>'
            f'<span style="flex:1; color:#F9FAFB; font-weight:600;">{item["id"]}</span>'
            f'<span style="color:#9CA3AF; margin-right:1rem;">${item["amount"]:.2f}</span>'
            f'<span style="color:#9CA3AF; margin-right:1rem;">{item["timestamp"]}</span>'
            f'<span style="color:{color}; font-weight:700; margin-right:0.5rem;">'
            f'{item["probability"]*100:.1f}%</span>'
            f'{risk_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

    if history:
        csv = pd.DataFrame(history).to_csv(index=False)
        st.download_button(
            "⬇️ Tải lịch sử CSV",
            csv,
            "history.csv",
            "text/csv",
            use_container_width=True,
        )


# ============================================================================
# Main
# ============================================================================

def main():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "random_transaction" not in st.session_state:
        st.session_state["random_transaction"] = None
    if "explain_data" not in st.session_state:
        st.session_state["explain_data"] = {}

    render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🔍 Kiểm tra GD",
        "📦 Batch Analysis",
        "🧠 Explain AI",
        "📜 Lịch sử",
    ])

    with tab1:
        render_dashboard()
    with tab2:
        render_prediction_form()
    with tab3:
        render_batch_tab()
    with tab4:
        render_explain_tab()
    with tab5:
        render_history_tab()


if __name__ == "__main__":
    main()
