import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def load_data(path="weatherAUS.csv"):
    df = pd.read_csv(path, na_values='NA')
    print(f"Đã load dữ liệu: {df.shape[0]} hàng, {df.shape[1]} cột")
    return df

def handle_missing(df):

    df = df.copy()

    if 'RainTomorrow' in df.columns:
        before = len(df)
        df = df.dropna(subset=['RainTomorrow'])  
        after = len(df)

    for col in ['RainToday', 'RainTomorrow']:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    if 'RainToday' in df.columns:
        df['RainToday'] = df['RainToday'].fillna(2)

    for col in df.select_dtypes(include=np.number).columns:
        if col not in ['RainToday', 'RainTomorrow']:
            df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include='object').columns:
        if col != 'Date':
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Xử lý missing hoàn tất")
    return df

def handle_outliers_iqr(df):
     # Chỉ lấy các cột số (trừ target)
    num_cols = df.select_dtypes(include=np.number).columns.drop(['RainToday', 'RainTomorrow'], errors='ignore')
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    print("Hoàn thành xử lý outliers bằng IQR")
    return df

def normalize_features(df):
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
    
    print(f"Đã xử lý cột {date_col}")
    return df

def label_encoding(df):
    le_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
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