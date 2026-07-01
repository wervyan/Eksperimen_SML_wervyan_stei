import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "bank_customer_churn_raw",
    "Bank Customer Churn Prediction.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "bank_customer_churn_preprocessing"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "bank_customer_churn_preprocessed.csv"
)

def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset mentah Bank Customer Churn.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File dataset tidak ditemukan: {path}")
    
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan preprocessing otomatis sesuai tahapan eksperimen manual.

    Tahapan:
    1. Menghapus kolom customer_id.
    2. Menghapus missing value.
    3. Menghapus data duplikat.
    4. Melakukan encoding fitur kategorikal.
    5. Melakukan scaling fitur numerik.
    6. Menggabungkan kembali fitur dan target.
    """
    df_clean = df.copy()

    target_col = "churn"

    if target_col not in df_clean.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan pada dataset.")
    
    if "customer_id" in df_clean.columns:
        df_clean = df_clean.drop(columns="customer_id")

    df_clean = df_clean.dropna()
    df_clean = df_clean.drop_duplicates()

    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        encoder = LabelEncoder()
        df_clean[col] = encoder.fit_transform(df_clean[col])
    
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    df_preprocessed = X_scaled.copy()
    df_preprocessed[target_col] = y.values

    return df_preprocessed

def save_data(df: pd.DataFrame, output_path:str) -> None:
    """
    Menyimpan dataset hasil preprocessing.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def main() -> None:
    print("Memuat dataset mentah...")
    df = load_data(RAW_DATA_PATH)

    print("Menjalankan preprocessing otomatis...")
    df_preprocessed = preprocess_data(df)

    print("Menyimpan dataset hasil preprocessing...")
    save_data(df_preprocessed, OUTPUT_PATH)

    print("Preprocessing selesai.")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Shape dataset hasil processing: {df_preprocessed.shape}")
    print(df_preprocessed.columns.tolist())

if __name__ == "__main__":
    main()