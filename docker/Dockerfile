# ============================================================================
# Dockerfile - Credit Card Fraud Detection
# ============================================================================
# Base image: Python 3.9-slim (nhẹ, đủ cho Spark + SystemDS)
# Yêu cầu: OpenJDK 11 (bắt buộc cho Apache Spark và SystemDS)

FROM python:3.9-slim

# ============================================================================
# Labels
# ============================================================================
LABEL maintainer="CreditCard Fraud Detection Team"
LABEL description="Credit Card Fraud Detection with Apache Spark, SystemDS, FastAPI & Streamlit"
LABEL version="1.0.0"

# ============================================================================
# Cài đặt hệ thống
# ============================================================================

# Cập nhật package manager và cài đặt các dependencies cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jdk \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Cấu hình Java
# ============================================================================
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Kiểm tra Java
RUN java -version

# ============================================================================
# Cài đặt Python dependencies
# ============================================================================

# Copy requirements file
COPY docker/requirements.txt /tmp/requirements.txt

# Cài đặt thư viện Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================================
# Thiết lập thư mục làm việc
# ============================================================================
WORKDIR /app

# Copy toàn bộ source code
COPY . /app

# ============================================================================
# Mặc định: chạy backend API (có thể override khi run container)
# ============================================================================
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
