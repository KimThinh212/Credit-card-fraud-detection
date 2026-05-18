#!/bin/bash
# ============================================================================
# Pipeline Script - Credit Card Fraud Detection
# ============================================================================
# Script chạy toàn bộ pipeline theo thứ tự:
#   1. Kiểm tra môi trường
#   2. Chạy Spark preprocessing (data_processing.py)
#   3. Chạy SystemDS training (train_systemds.py)
#   4. Build & start Docker Compose (FastAPI + Streamlit)
# ============================================================================

set -e  # Dừng script nếu có lỗi

# ============================================================================
# Màu sắc cho terminal output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Helper functions
# ============================================================================
print_step() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  📌 $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}  ✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}  ⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}  ❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}  ℹ️  $1${NC}"
}

# ============================================================================
# Bước 0: Kiểm tra môi trường
# ============================================================================
print_step "Bước 0: Kiểm tra môi trường"

# Kiểm tra Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
    print_success "Python: $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON=python
    print_success "Python: $(python --version)"
else
    print_error "Python chưa được cài đặt!"
    exit 1
fi

# Kiểm tra Java (bắt buộc cho Spark & SystemDS)
if command -v java &> /dev/null; then
    print_success "Java: $(java -version 2>&1 | head -n 1)"
else
    print_warning "Java chưa được cài đặt! Spark & SystemDS sẽ không chạy được."
    print_info "Cài đặt: sudo apt install openjdk-11-jdk (Ubuntu) hoặc dùng Docker."
fi

# Kiểm tra JAVA_HOME
if [ -z "$JAVA_HOME" ]; then
    print_warning "JAVA_HOME chưa được set. Spark có thể gặp lỗi."
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
    print_info "Tạm thời set JAVA_HOME=$JAVA_HOME"
fi

# Kiểm tra Docker & Docker Compose
if command -v docker &> /dev/null; then
    print_success "Docker: $(docker --version)"
    if docker compose version &> /dev/null; then
        print_success "Docker Compose: $(docker compose version)"
    elif command -v docker-compose &> /dev/null; then
        print_success "Docker Compose: $(docker-compose --version)"
        COMPOSE_CMD="docker-compose"
    else
        print_warning "Docker Compose chưa được cài đặt!"
        COMPOSE_CMD=""
    fi
else
    print_warning "Docker chưa được cài đặt!"
    COMPOSE_CMD=""
fi

# Kiểm tra file dữ liệu
DATA_FILE="data/raw/creditcard.csv"
if [ -f "$DATA_FILE" ]; then
    print_success "Dữ liệu: $DATA_FILE"
else
    print_error "Không tìm thấy file dữ liệu: $DATA_FILE"
    print_info "Vui lòng tải dataset từ: https://www.kaggle.com/mlg-ulb/creditcardfraud"
    print_info "và đặt vào thư mục: $DATA_FILE"
    exit 1
fi

# ============================================================================
# Bước 1: Cài đặt môi trường ảo (tùy chọn)
# ============================================================================
print_step "Bước 1: Thiết lập môi trường Python"

VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    print_info "Đang tạo môi trường ảo Python..."
    $PYTHON -m venv $VENV_DIR
    print_success "Môi trường ảo đã được tạo."
else
    print_info "Môi trường ảo đã tồn tại, bỏ qua."
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null
print_success "Môi trường ảo đã được kích hoạt."

# Cài đặt dependencies
print_info "Đang cài đặt các thư viện Python..."
pip install --quiet --upgrade pip
pip install --quiet -r docker/requirements.txt
print_success "Đã cài đặt tất cả thư viện."

# ============================================================================
# Bước 2: Chạy Spark Preprocessing
# ============================================================================
print_step "Bước 2: Xử lý dữ liệu với Apache Spark"

print_info "Chạy: $PYTHON src/data_processing.py"
echo ""
$PYTHON src/data_processing.py

if [ $? -eq 0 ]; then
    print_success "Xử lý dữ liệu Spark hoàn tất!"
else
    print_error "Xử lý dữ liệu Spark thất bại!"
    exit 1
fi

# ============================================================================
# Bước 3: Chạy SystemDS Training
# ============================================================================
print_step "Bước 3: Huấn luyện mô hình với Apache SystemDS"

print_info "Chạy: $PYTHON src/train_systemds.py"
echo ""
$PYTHON src/train_systemds.py

if [ $? -eq 0 ]; then
    print_success "Huấn luyện SystemDS hoàn tất!"
else
    print_error "Huấn luyện SystemDS thất bại!"
    exit 1
fi

# Kiểm tra file weights đã được tạo
if [ -f "data/processed/model_weights.csv" ]; then
    print_success "Model weights: data/processed/model_weights.csv"
else
    print_error "Model weights không được tạo!"
    exit 1
fi

# ============================================================================
# Bước 4: Build & Start Docker Compose
# ============================================================================
print_step "Bước 4: Build và khởi động Docker Compose"

if [ -n "$COMPOSE_CMD" ]; then
    print_info "Đang build Docker images..."
    docker compose build

    print_info "Đang khởi động services..."
    docker compose up -d

    print_success "Docker Compose đã khởi động!"
else
    print_warning "Docker Compose không khả dụng. Chạy services thủ công:"
    echo ""
    echo -e "  ${CYAN}Terminal 1 - Backend API:${NC}"
    echo "  uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload"
    echo ""
    echo -e "  ${CYAN}Terminal 2 - Frontend UI:${NC}"
    echo "  streamlit run src/web_ui.py --server.port=8501 --server.address=0.0.0.0"
fi

# ============================================================================
# Hoàn tất
# ============================================================================
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           🎉 PIPELINE HOÀN TẤT THÀNH CÔNG! 🎉                ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}📊 Frontend (Streamlit):${NC}  http://localhost:8501"
echo -e "  ${CYAN}🔌 Backend API (FastAPI):${NC} http://localhost:8000"
echo -e "  ${CYAN}📖 API Docs:${NC}             http://localhost:8000/docs"
echo -e "  ${CYAN}🔍 Health Check:${NC}         http://localhost:8000/health"
echo ""
echo -e "  ${YELLOW}Để dừng services: docker compose down${NC}"
echo ""

# Deactivate virtual environment
deactivate 2>/dev/null || true
