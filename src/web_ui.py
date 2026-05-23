"""
Streamlit Dashboard - Credit Card Fraud Detection
Chức năng:
  - Dashboard tổng quan với biểu đồ Plotly
  - Form dự đoán realtime
  - Random transaction generator
  - Hiển thị kết quả với progress bar và hiệu ứng
"""

import os
import sys
import json
import random
from typing import Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    /* Tổng thể */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4A6A8A;
        margin-bottom: 2rem;
    }

    /* Card */
    .card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border: 1px solid #E8EDF2;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }

    /* Metric Card tùy chỉnh */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
    }
    .metric-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card.red {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
    }
    .metric-card.orange {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.2rem;
    }

    /* Alert colors */
    .alert-low {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 600;
    }
    .alert-medium {
        background: linear-gradient(135deg, #f2994a, #f2c94c);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 600;
    }
    .alert-high {
        background: linear-gradient(135deg, #cb2d3e, #ef473a);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 600;
    }

    /* Button */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* Form chi tiết */
    div[data-testid="column"] {
        padding: 0 5px;
    }
    .feature-input label {
        font-size: 0.8rem !important;
        color: #4A6A8A !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Constants
# ============================================================================

API_URL = "http://localhost:8000"
DEFAULT_FEATURE_MEAN = 0.0
DEFAULT_FEATURE_STD = 1.0
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)]

# Các giá trị thống kê mô phỏng cho V features (PCA components ~ N(0,1))
V_STATS = {f"V{i}": {"mean": 0.0, "std": 1.0} for i in range(1, 29)}


# ============================================================================
# Helper functions
# ============================================================================

@st.cache_data(ttl=300)
def get_dashboard_data():
    """
    Tạo dữ liệu giả lập cho dashboard.
    Dùng để vẽ biểu đồ khi chưa có dữ liệu thật.
    """
    np.random.seed(42)

    n_transactions = 1000

    # Mô phỏng Amount (phân phối lệch phải - giống thực tế)
    amounts = np.random.exponential(scale=100, size=n_transactions)
    amounts = np.clip(amounts, 1, 2000)

    # Mô phỏng Time
    times = np.random.uniform(0, 172792, size=n_transactions)

    # Mô phỏng Class
    fraud_ratio = 0.017  # ~1.7% fraud (undersampled real ratio)
    classes = np.random.choice([0, 1], size=n_transactions, p=[1-fraud_ratio, fraud_ratio])

    # Tạo V features (PCA components ~ N(0,1))
    v_features = {}
    for i in range(1, 29):
        # Fraud transactions có xu hướng có V features khác biệt
        v_mean = 0.5 if i in [3, 4, 9, 10, 11, 12, 14, 16, 17] else 0.0
        v_features[f"V{i}"] = np.random.normal(
            loc=np.where(classes == 1, v_mean, 0.0),
            scale=1.0,
            size=n_transactions
        )

    df = pd.DataFrame({
        "Time": times,
        "Amount": amounts,
        "Class": classes,
        **v_features
    })

    # Thêm một số outlier cho fraud
    fraud_mask = df["Class"] == 1
    df.loc[fraud_mask, "Amount"] = df.loc[fraud_mask, "Amount"] * 2.5

    return df


def generate_random_transaction():
    """
    Sinh giao dịch ngẫu nhiên để demo.
    - 90%: giao dịch hợp lệ (V features ngẫu nhiên ~ N(0,1))
    - 10%: giao dịch gian lận (V features có độ lệch)
    """
    is_fraud_sim = random.random() < 0.1

    transaction = {}

    # V1-V28: PCA components
    for i in range(1, 29):
        if is_fraud_sim and i in [3, 4, 9, 10, 11, 12, 14, 16, 17]:
            # Các V feature thường có giá trị lớn hơn trong fraud
            transaction[f"V{i}"] = round(random.gauss(-1.5, 1.2), 6)
        else:
            transaction[f"V{i}"] = round(random.gauss(0.0, 1.0), 6)

    # Time
    transaction["Time"] = round(random.uniform(0, 172792), 2)

    # Amount
    if is_fraud_sim:
        transaction["Amount"] = round(random.uniform(50, 2000), 2)
    else:
        transaction["Amount"] = round(random.uniform(1, 300), 2)

    return transaction


def call_predict_api(transaction: dict) -> Optional[dict]:
    """
    Gọi API dự đoán
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API lỗi: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Không thể kết nối đến API tại {API_URL}. Đảm bảo API đang chạy.")
        return None
    except Exception as e:
        st.error(f"Lỗi kết nối API: {e}")
        return None


# ============================================================================
# Dashboard - Biểu đồ
# ============================================================================

def render_dashboard():
    """
    Render Dashboard tổng quan với biểu đồ Plotly
    """
    st.markdown('<div class="main-header">📊 Tổng quan Hệ thống</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Phân tích giao dịch & Phát hiện gian lận thời gian thực</div>',
                unsafe_allow_html=True)

    # Load dữ liệu
    df = get_dashboard_data()

    # ---- Metrics Row ----
    total_tx = len(df)
    fraud_tx = int(df["Class"].sum())
    legit_tx = total_tx - fraud_tx
    fraud_ratio = fraud_tx / total_tx * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{total_tx:,}</div>
                <div class="metric-label">Tổng giao dịch đang quét</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card green">
                <div class="metric-value">{legit_tx:,}</div>
                <div class="metric-label">Giao dịch an toàn</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card red">
                <div class="metric-value">{fraud_tx:,}</div>
                <div class="metric-label">Gian lận phát hiện</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col4:
        color_class = "red" if fraud_ratio > 3 else "orange"
        st.markdown(
            f"""<div class="metric-card {color_class}">
                <div class="metric-value">{fraud_ratio:.2f}%</div>
                <div class="metric-label">Tỷ lệ rủi ro</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Charts Row ----
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">'
                     '<div class="card-title">📈 Phân phối Số tiền Giao dịch</div>',
                     unsafe_allow_html=True)

        fig_amount = px.histogram(
            df, x="Amount", color="Class",
            color_discrete_map={0: "#4CAF50", 1: "#F44336"},
            nbins=50,
            labels={"Amount": "Số tiền (USD)", "Class": "Loại giao dịch", "count": "Số lượng"},
            title="",
            opacity=0.75
        )
        fig_amount.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=30),
            legend=dict(
                title="Loại",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(range=[0, 500])
        )
        st.plotly_chart(fig_amount, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">'
                     '<div class="card-title">📊 So sánh Giao dịch An toàn vs Gian lận</div>',
                     unsafe_allow_html=True)

        class_counts = df["Class"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        class_counts["Label"] = class_counts["Class"].map({0: "An toàn", 1: "Gian lận"})

        fig_bar = px.bar(
            class_counts,
            x="Label",
            y="Count",
            color="Label",
            color_discrete_map={"An toàn": "#4CAF50", "Gian lận": "#F44336"},
            text="Count",
            labels={"Count": "Số lượng", "Label": ""},
            title=""
        )
        fig_bar.update_traces(textposition="outside", textfont=dict(size=16, weight="bold"))
        fig_bar.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=30),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Second Row ----
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">'
                     '<div class="card-title">🕐 Phân phối Thời gian Giao dịch</div>',
                     unsafe_allow_html=True)

        fig_time = px.histogram(
            df, x="Time", color="Class",
            color_discrete_map={0: "#2196F3", 1: "#F44336"},
            nbins=50,
            labels={"Time": "Thời gian (giây)", "Class": "Loại giao dịch", "count": "Số lượng"},
            title="",
            opacity=0.75
        )
        fig_time.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=30),
            legend=dict(
                title="Loại",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">'
                     '<div class="card-title">🎯 Tỷ lệ Phát hiện Gian lận</div>',
                     unsafe_allow_html=True)

        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=["An toàn", "Gian lận"],
            values=[legit_tx, fraud_tx],
            marker=dict(colors=["#4CAF50", "#F44336"]),
            textinfo="label+percent",
            hole=0.4
        ))
        fig_pie.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=30),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# Realtime Fraud Detection Form
# ============================================================================

def render_prediction_form():
    """
    Render form dự đoán realtime
    """
    st.markdown('<div class="main-header">🔍 Kiểm tra Giao dịch</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Nhập thông tin giao dịch để kiểm tra gian lận</div>',
                unsafe_allow_html=True)

    # Form chính
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Hàng 1: Amount và Time
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input(
                "💰 Số tiền (Amount)",
                min_value=0.0,
                max_value=50000.0,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Số tiền của giao dịch tính bằng USD"
            )
        with col2:
            time_val = st.number_input(
                "⏱ Thời gian (Time)",
                min_value=0.0,
                max_value=200000.0,
                value=75000.0,
                step=1.0,
                format="%.2f",
                help="Thời gian tính bằng giây kể từ giao dịch đầu tiên"
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Các V features - chia thành nhiều hàng, mỗi hàng 4 cột
        st.markdown('<div class="card-title">🔬 PCA Components (V1-V28)</div>',
                    unsafe_allow_html=True)
        st.caption("Các thành phần PCA đã được giảm chiều từ dữ liệu gốc. "
                    "Giá trị thường nằm trong khoảng [-5, 5].")

        v_values = {}
        # 7 rows x 4 cols = 28 features
        for row_idx in range(7):
            cols = st.columns(4)
            for col_idx in range(4):
                feat_idx = row_idx * 4 + col_idx + 1
                if feat_idx <= 28:
                    feat_name = f"V{feat_idx}"
                    with cols[col_idx]:
                        v_values[feat_name] = st.number_input(
                            feat_name,
                            min_value=-10.0,
                            max_value=10.0,
                            value=0.0,
                            step=0.01,
                            format="%.4f",
                            key=f"v_{feat_idx}",
                            help=f"PCA component {feat_name}"
                        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Các nút chức năng
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        random_btn = st.button("🎲 Random Transaction", use_container_width=True)
    with col2:
        predict_btn = st.button("🚀 Kiểm tra", type="primary", use_container_width=True)
    with col3:
        st.markdown("")  # Spacer

    # Xử lý các nút
    if random_btn:
        rand_data = generate_random_transaction()
        # Điền vào form bằng session_state
        st.session_state["random_transaction"] = rand_data
        st.rerun()

    if "random_transaction" in st.session_state:
        # Đã có dữ liệu từ nút Random Transaction
        rt = st.session_state["random_transaction"]
        amount = rt["Amount"]
        time_val = rt["Time"]
        for k, v in rt.items():
            if k.startswith("V"):
                v_values[k] = v
        st.info(f"🎲 Đã sinh giao dịch ngẫu nhiên: Amount=${amount:.2f}, Time={time_val:.0f}s")

    # Thực hiện dự đoán
    if predict_btn:
        transaction = {
            **{f"V{i}": v_values.get(f"V{i}", 0.0) for i in range(1, 29)},
            "Time": time_val,
            "Amount": amount
        }

        with st.spinner("🔄 Đang phân tích giao dịch..."):
            result = call_predict_api(transaction)

        if result:
            is_fraud = result["is_fraud"]
            probability = result["fraud_probability"]
            risk_level = result["risk_level"]

            # Hiển thị kết quả
            st.markdown("---")
            st.markdown("### 📋 Kết quả phân tích")

            # Progress bar thể hiện mức độ rủi ro
            prob_pct = probability * 100
            st.markdown(f"**Xác suất gian lận:** {prob_pct:.2f}%")

            progress_color = (
                "green" if risk_level == "Low"
                else "orange" if risk_level == "Medium"
                else "red"
            )
            st.progress(int(prob_pct), text=f"Mức rủi ro: {risk_level}")

            # Hiển thị alert
            if risk_level == "Low":
                st.markdown(
                    f"""<div class="alert-low">
                        ✅ Giao dịch AN TOÀN (Xác suất: {prob_pct:.2f}%)
                    </div>""",
                    unsafe_allow_html=True
                )
                st.success("Giao dịch này có vẻ an toàn!")
            elif risk_level == "Medium":
                st.markdown(
                    f"""<div class="alert-medium">
                        ⚠️ Giao dịch CẦN XEM XÉT (Xác suất: {prob_pct:.2f}%)
                    </div>""",
                    unsafe_allow_html=True
                )
                st.warning("Giao dịch này cần được xem xét thêm!")
            else:
                st.markdown(
                    f"""<div class="alert-high">
                        🚨 CẢNH BÁO GIAN LẬN (Xác suất: {prob_pct:.2f}%)
                    </div>""",
                    unsafe_allow_html=True
                )
                st.error("Phát hiện giao dịch gian lận!")

            # Chi tiết
            with st.expander("🔎 Xem chi tiết phân tích"):
                st.json({
                    "is_fraud": is_fraud,
                    "fraud_probability": probability,
                    "risk_level": risk_level,
                    "input": {
                        "Amount": amount,
                        "Time": time_val,
                        "V_features": {k: v for k, v in transaction.items() if k.startswith("V")}
                    }
                })


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """
    Render sidebar với thông tin hệ thống
    """
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/credit-card.png", width=80)
        st.markdown("## 💳 Fraud Detection")
        st.markdown("---")

        st.markdown("### Trạng thái hệ thống")
        try:
            health = requests.get(f"{API_URL}/health", timeout=5)
            if health.status_code == 200:
                st.success("✅ API đang hoạt động")
            else:
                st.warning("⚠️ API có vấn đề")
        except:
            st.error("❌ API không khả dụng")

        st.markdown("### Thông tin")
        st.info(
            """
            **Dataset:** Credit Card Fraud Detection  
            **Model:** Logistic Regression (SystemDS)  
            **Features:** 30 (V1-V28 + Time + Amount)  
            **Framework:** Apache Spark + SystemDS
            """
        )

        st.markdown("### Hướng dẫn")
        st.caption(
            """
            1. Xem dashboard tổng quan
            2. Nhập thông tin giao dịch
            3. Bấm "Kiểm tra" để dự đoán
            """
        )

        st.markdown("---")
        st.caption("Credit Card Fraud Detection v1.0")


# ============================================================================
# Main
# ============================================================================

def main():
    """
    Hàm chính - Điều phối các thành phần
    """
    render_sidebar()

    # Tabs
    tab1, tab2 = st.tabs(["📊 Dashboard", "🔍 Kiểm tra Giao dịch"])

    with tab1:
        render_dashboard()

    with tab2:
        render_prediction_form()


if __name__ == "__main__":
    main()
