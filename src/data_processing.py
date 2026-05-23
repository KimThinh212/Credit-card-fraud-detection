"""
Xử lý dữ liệu Credit Card Fraud Detection sử dụng Apache Spark
Chức năng:
  - Khởi tạo SparkSession với cấu hình tối ưu bộ nhớ
  - Đọc và làm sạch dữ liệu
  - Chuẩn hóa các cột Time và Amount
  - Xử lý mất cân bằng dữ liệu (undersampling)
  - Chia Train/Test và lưu kết quả
"""

import os
import sys
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Thêm thư mục gốc vào sys.path để import module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, rand
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array


def create_spark_session():
    """
    Khởi tạo SparkSession với cấu hình tối ưu bộ nhớ
    """
    spark = SparkSession.builder \
        .appName("CreditCardFraudDetection") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession đã được khởi tạo thành công.")
    return spark


def load_and_clean_data(spark, raw_path):
    """
    Đọc dữ liệu từ CSV và làm sạch (xử lý null, đúng kiểu dữ liệu)
    """
    logger.info(f"Đang đọc dữ liệu từ {raw_path}...")

    df = spark.read.csv(raw_path, header=True, inferSchema=True)

    # Log số lượng bản ghi và cột
    logger.info(f"Số lượng bản ghi: {df.count()}")
    logger.info(f"Số lượng cột: {len(df.columns)}")

    # Kiểm tra giá trị null
    null_counts = df.select([
        count(when(col(c).isNull(), c)).alias(c) for c in df.columns
    ])
    logger.info("Giá trị null theo cột:")
    null_counts.show(truncate=False)

    # Xử lý null: drop các dòng có null (nếu có)
    initial_count = df.count()
    df = df.dropna()
    after_drop_count = df.count()
    if initial_count > after_drop_count:
        logger.warning(f"Đã loại bỏ {initial_count - after_drop_count} dòng có giá trị null.")
    else:
        logger.info("Không có giá trị null trong dữ liệu.")

    # Đảm bảo cột Class là IntegerType
    df = df.withColumn("Class", col("Class").cast(IntegerType()))
    df = df.withColumn("Time", col("Time").cast(DoubleType()))
    df = df.withColumn("Amount", col("Amount").cast(DoubleType()))

    return df


def scale_features(df):
    """
    Chuẩn hóa các cột Time và Amount sử dụng VectorAssembler và StandardScaler
    """
    logger.info("Đang chuẩn hóa cột Time và Amount...")

    # Tạo vector từ Time và Amount
    assembler = VectorAssembler(
        inputCols=["Time", "Amount"],
        outputCol="time_amount_vec"
    )

    # Chuẩn hóa (StandardScaler: mean=0, std=1)
    scaler = StandardScaler(
        inputCol="time_amount_vec",
        outputCol="scaled_time_amount",
        withStd=True,
        withMean=True
    )

    # Pipeline xử lý
    pipeline = Pipeline(stages=[assembler, scaler])
    pipeline_model = pipeline.fit(df)
    scaled_df = pipeline_model.transform(df)

    # Chuyển vector sparse/dense thành array rồi tách thành các cột riêng
    scaled_df = scaled_df.withColumn(
        "scaled_array", vector_to_array(col("scaled_time_amount"))
    )
    scaled_df = scaled_df.withColumn("Time_scaled", col("scaled_array")[0])
    scaled_df = scaled_df.withColumn("Amount_scaled", col("scaled_array")[1])

    logger.info("Chuẩn hóa hoàn tất.")
    return scaled_df


def prepare_features(scaled_df):
    """
    Chuẩn bị feature matrix: chọn cột V1-V28 và Time_scaled, Amount_scaled
    """
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]
    label_col = "Class"

    feature_df = scaled_df.select(feature_cols + [label_col])
    return feature_df, feature_cols, label_col


def balance_data(feature_df, label_col, seed=42, sampling_ratio=10):
    """
    Xử lý mất cân bằng dữ liệu bằng phương pháp stratified sampling
    - Lấy TẤT CẢ các mẫu của class thiểu số (fraud = 1)
    - Random sample từ class đa số (legit = 0) với tỷ lệ sampling_ratio:1
      (sampling_ratio legit cho mỗi 1 fraud, giúp tăng lượng dữ liệu training)
    """
    logger.info("Đang xử lý mất cân bằng dữ liệu...")

    # Đếm số lượng từng class
    class_counts = feature_df.groupBy(label_col).agg(count("*").alias("count")).collect()
    class_dict = {row[label_col]: row["count"] for row in class_counts}

    legit_count = class_dict.get(0, 0)
    fraud_count = class_dict.get(1, 0)

    logger.info(f"Số lượng giao dịch hợp lệ (Class=0): {legit_count}")
    logger.info(f"Số lượng giao dịch gian lận (Class=1): {fraud_count}")
    logger.info(f"Tỷ lệ mất cân bằng: 1:{legit_count // max(fraud_count, 1)}")

    # Tách class
    fraud_df = feature_df.filter(col(label_col) == 1)
    legit_df = feature_df.filter(col(label_col) == 0)

    # Stratified sampling: giữ ALL fraud, sample legit với tỷ lệ sampling_ratio:1
    target_legit_count = fraud_count * sampling_ratio
    if target_legit_count > legit_count:
        target_legit_count = legit_count

    sample_fraction = target_legit_count / legit_count
    legit_sampled = legit_df.sample(False, sample_fraction, seed=seed)

    # Gộp lại
    balanced_df = legit_sampled.union(fraud_df)

    # Xáo trộn dữ liệu
    balanced_df = balanced_df.orderBy(rand(seed))

    final_legit = balanced_df.filter(col(label_col) == 0).count()
    final_fraud = balanced_df.filter(col(label_col) == 1).count()
    logger.info(f"Sau stratified sampling (ratio={sampling_ratio}:1) - Hợp lệ: {final_legit}, Gian lận: {final_fraud}")
    logger.info(f"Tổng số mẫu training: {final_legit + final_fraud}")
    logger.info(f"Dữ liệu đã được cân bằng.")

    return balanced_df


def save_train_test(balanced_df, feature_cols, label_col, output_dir, seed=42):
    """
    Chia dữ liệu thành Train (80%) và Test (20%), lưu dưới dạng CSV
    """
    logger.info("Đang chia Train/Test...")

    train_df, test_df = balanced_df.randomSplit([0.8, 0.2], seed=seed)

    train_count = train_df.count()
    test_count = test_df.count()
    logger.info(f"Train: {train_count} mẫu, Test: {test_count} mẫu")

    # Tách features và labels
    X_train = train_df.select(feature_cols)
    y_train = train_df.select(label_col)
    X_test = test_df.select(feature_cols)
    y_test = test_df.select(label_col)

    # Đảm bảo thư mục output tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lưu thành CSV (có header, không có index)
    # SystemDS hỗ trợ đọc CSV trực tiếp
    X_train.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        os.path.join(output_dir, "X_train_tmp")
    )
    y_train.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        os.path.join(output_dir, "y_train_tmp")
    )
    X_test.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        os.path.join(output_dir, "X_test_tmp")
    )
    y_test.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        os.path.join(output_dir, "y_test_tmp")
    )

    # Spark ghi CSV trong thư mục, cần copy file part ra ngoài
    import shutil
    import glob

    def move_csv(tmp_dir, target_file):
        pattern = os.path.join(tmp_dir, "part-*.csv")
        csv_files = glob.glob(pattern)
        if csv_files:
            shutil.copy(csv_files[0], target_file)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"Đã lưu: {target_file}")

    move_csv(
        os.path.join(output_dir, "X_train_tmp"),
        os.path.join(output_dir, "X_train.csv")
    )
    move_csv(
        os.path.join(output_dir, "y_train_tmp"),
        os.path.join(output_dir, "y_train.csv")
    )
    move_csv(
        os.path.join(output_dir, "X_test_tmp"),
        os.path.join(output_dir, "X_test.csv")
    )
    move_csv(
        os.path.join(output_dir, "y_test_tmp"),
        os.path.join(output_dir, "y_test.csv")
    )

    logger.info("Đã lưu toàn bộ dữ liệu Train/Test thành công.")
    return X_train, y_train, X_test, y_test


def process_and_save_balanced(raw_df, feature_cols, label_col, output_balanced_dir, seed=42):
    """
    Lưu dữ liệu đã được làm sạch + chuẩn hóa (chưa undersample)
    để phục vụ dashboard Streamlit (hiển thị phân phối thật)
    """
    os.makedirs(output_balanced_dir, exist_ok=True)
    # Lưu full dataset đã scale để dashboard dùng
    out_df = raw_df.select(feature_cols + [label_col])
    out_path = os.path.join(output_balanced_dir, "full_scaled.csv")
    out_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        os.path.join(output_balanced_dir, "full_scaled_tmp")
    )
    import shutil, glob
    pattern = os.path.join(output_balanced_dir, "full_scaled_tmp", "part-*.csv")
    csv_files = glob.glob(pattern)
    if csv_files:
        shutil.copy(csv_files[0], out_path)
        shutil.rmtree(os.path.join(output_balanced_dir, "full_scaled_tmp"), ignore_errors=True)
        logger.info(f"Đã lưu dữ liệu full scale: {out_path}")


def main():
    """
    Pipeline chính: load -> clean -> scale -> balance -> save
    """
    raw_path = os.path.join(project_root, "data", "raw", "creditcard.csv")
    output_dir = os.path.join(project_root, "data", "processed")
    output_balanced_dir = os.path.join(project_root, "data", "processed")

    # Kiểm tra file đầu vào
    if not os.path.exists(raw_path):
        logger.error(f"Không tìm thấy file dữ liệu: {raw_path}")
        logger.error("Vui lòng tải dataset từ https://www.kaggle.com/mlg-ulb/creditcardfraud")
        sys.exit(1)

    # Khởi tạo Spark
    spark = create_spark_session()

    try:
        # Bước 1: Đọc và làm sạch
        df = load_and_clean_data(spark, raw_path)

        # Bước 2: Chuẩn hóa Time và Amount
        scaled_df = scale_features(df)

        # Bước 3: Chuẩn bị feature columns
        feature_df, feature_cols, label_col = prepare_features(scaled_df)

        # Lưu full data đã scale (phục vụ dashboard)
        process_and_save_balanced(scaled_df, feature_cols, label_col, output_balanced_dir)

        # Bước 4: Xử lý mất cân bằng (stratified sampling, ratio 10:1 legit:fraud)
        balanced_df = balance_data(feature_df, label_col, sampling_ratio=10)

        # Bước 5: Chia Train/Test và lưu
        save_train_test(balanced_df, feature_cols, label_col, output_dir)

        logger.info("Pipeline xử lý dữ liệu hoàn tất!")

    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý: {e}")
        raise
    finally:
        spark.stop()
        logger.info("SparkSession đã đóng.")


if __name__ == "__main__":
    main()
