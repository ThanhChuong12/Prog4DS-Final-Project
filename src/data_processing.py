import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_data(path="weatherAUS.csv"):
    df = pd.read_csv(path, na_values='NA')
    print(f"Đã load dữ liệu: {df.shape[0]} hàng, {df.shape[1]} cột")
    return df

def handle_missing(df):

    df = df.copy()

    # 1. XỬ LÝ BIẾN MỤC TIÊU (Bắt buộc xóa NA)
    if 'RainTomorrow' in df.columns:
        df = df.dropna(subset=['RainTomorrow'])

    # 2. MÃ HÓA BIẾN PHÂN LOẠI NHỊ PHÂN (RainToday/Tomorrow)
    for col in ['RainToday', 'RainTomorrow']:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    if 'RainToday' in df.columns:
        df['RainToday'] = df['RainToday'].fillna(2) # 2 là trạng thái "Không xác định"

    # 3. TẠO MISSING INDICATOR FLAGS (Cho nhóm biến tỷ lệ thiếu cao)
    # Giúp mô hình nhận biết trạng thái "không quan trắc được"
    high_miss_cols = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']
    for col in high_miss_cols:
        if col in df.columns:
            df[f'{col}_Missing'] = df[col].isnull().astype(int)

    # 4. XỬ LÝ BIẾN SỐ BẰNG ITERATIVE IMPUTER (MICE)
    # Lấy danh sách các cột số (trừ target và các flag vừa tạo)
    num_cols = df.select_dtypes(include=np.number).columns.drop(['RainToday', 'RainTomorrow'], errors='ignore')
    num_cols = [c for c in num_cols if not c.endswith('_Missing')]

    # Khởi tạo MICE Imputer
    # ItrativeImputer sẽ dự đoán giá trị thiếu dựa trên các biến tương quan (nhiệt độ, áp suất...)
    mice_imputer = IterativeImputer(max_iter=20, random_state=42, tol=1e-2)
    
    # Thực hiện điền giá trị
    df[num_cols] = mice_imputer.fit_transform(df[num_cols])

    # 5. XỬ LÝ BIẾN PHÂN LOẠI CÒN LẠI (Mode Imputation)
    for col in df.select_dtypes(include='object').columns:
        if col != 'Date':
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Xử lý missing hoàn tất")
    return df



def normalize_features(df):

    #Loại bỏ số âm phát sinh do Imputer trước khi Log
    df['Rainfall'] = df['Rainfall'].clip(lower=0)
    # Rainfall: log + robust
    df['Rainfall'] = np.log1p(df['Rainfall'])
    df['Rainfall'] = RobustScaler().fit_transform(df[['Rainfall']])

    # Temp & Humidity: StandardScaler
    temp_hum_cols = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'Humidity9am', 'Humidity3pm']
    df[temp_hum_cols] = StandardScaler().fit_transform(df[temp_hum_cols])

    # Các cột còn lại: MinMaxScaler
    minmax_cols = ['Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 
                   'WindSpeed3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm']
    df[minmax_cols] = MinMaxScaler().fit_transform(df[minmax_cols])

    print("Hoàn thành chuẩn hóa dữ liệu")
    return df

def parse_date_column(df, date_col='Date', format='%Y-%m-%d'):
    df[date_col] = pd.to_datetime(df[date_col], format=format, errors='coerce')
    df['Month'] = df[date_col].dt.month
    df['Year'] = df[date_col].dt.year
    def get_season(month):
        if month in [12, 1, 2]: return 'Summer'
        elif month in [3, 4, 5]: return 'Autumn'
        elif month in [6, 7, 8]: return 'Winter'
        else: return 'Spring'
    df['Season'] = df['Month'].apply(get_season)
    print(f"Đã xử lý cột {date_col}")
    return df

def label_encoding(df):
    le_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']#'Location'
    le = LabelEncoder()
    for col in le_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print("Hoàn thành Label Encoding")
    return df

def pca_reduction(df, variance=0.95):
    # Chỉ lấy các cột số (trừ target)
    feature_cols = df.select_dtypes(include=np.number).columns.drop(['RainToday', 'RainTomorrow'], errors='ignore')
    X = df[feature_cols]

    pca = PCA(n_components=variance, random_state=42)
    X_pca = pca.fit_transform(X)

    pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

    final_df = pd.concat([
        df[['Date', 'RainToday', 'RainTomorrow']].reset_index(drop=True),
        df_pca
    ], axis=1)

    print(f"Giảm chiều thành công: {X_pca.shape[1]} components (giữ {variance*100}% variance)")
    return final_df

def train_test_split(df, test_size=0.2, random_state=42):
    
    np.random.seed(random_state)
    n = len(df)
    idx = np.random.permutation(n)
    test_cut = int(n * test_size)       
    
    test_df  = df.iloc[idx[:test_cut]].copy()
    train_df = df.iloc[idx[test_cut:]].copy()  
    print("Chia dữ liệu hoàn tất")
    return train_df, test_df