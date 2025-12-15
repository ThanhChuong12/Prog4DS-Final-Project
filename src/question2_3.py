import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def analyze_rainfall_distribution(df):
    """
    Câu 1: Vẽ Heatmap phân bố mưa (Rainfall đã scale) theo Mùa và Vị trí.
    """
    print("\n" + "="*40)
    print("1. PHÂN BỐ MƯA THEO MÙA VÀ VỊ TRÍ")
    print("="*40)
    
    if 'Location' not in df.columns or 'Rainfall' not in df.columns:
        print("Thiếu cột Location hoặc Rainfall.")
        return

    rainfall_pivot = df.pivot_table(index='Location', 
                                    columns='Season', 
                                    values='Rainfall', 
                                    aggfunc='mean')
    
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    rainfall_pivot = rainfall_pivot.reindex(columns=season_order)

    plt.figure(figsize=(14, 8))
    sns.heatmap(rainfall_pivot, cmap='YlGnBu', annot=False, linewidths=0.5)
    plt.title('Heatmap: Phân bố Lượng mưa (Scaled) theo Vị trí và Mùa')
    plt.ylabel('Location')
    plt.xlabel('Season')
    plt.show()


def test_pressure_hypothesis(df, location_name='Albury'):
    """
    Câu 2: Kiểm định quan hệ giữa Delta Pressure và RainTomorrow tại Albury.
    """
    print("\n" + "="*40)
    print(f"2. KIỂM ĐỊNH THỐNG KÊ TẠI {location_name.upper()}")
    print("="*40)

    df_loc = df[df['Location'] == location_name].copy()
    if df_loc.empty:
        print(f"Không tìm thấy dữ liệu cho {location_name}")
        return

    if 'Pressure3pm' in df_loc.columns and 'Pressure9am' in df_loc.columns:
        df_loc['DeltaPressure'] = df_loc['Pressure3pm'] - df_loc['Pressure9am']
    else:
        print("Dữ liệu thiếu cột Pressure.")
        return

    group_rain = df_loc[df_loc['RainTomorrow'] == 1]['DeltaPressure'].dropna()
    group_no_rain = df_loc[df_loc['RainTomorrow'] == 0]['DeltaPressure'].dropna()

    print(f"- Số mẫu nhóm Mưa (1): {len(group_rain)}")
    print(f"- Số mẫu nhóm Không Mưa (0): {len(group_no_rain)}")

    plt.figure(figsize=(10, 5))
    sns.kdeplot(group_rain, label='Mưa (RainTomorrow=1)', fill=True, color='blue')
    sns.kdeplot(group_no_rain, label='Không mưa (RainTomorrow=0)', fill=True, color='orange')
    plt.title(f'Phân phối Delta Pressure tại {location_name}')
    plt.xlabel('Delta Pressure (Pressure3pm - Pressure9am)')
    plt.legend()
    plt.show() 

    t_stat, p_val = stats.ttest_ind(group_rain, group_no_rain, equal_var=False)

    print("KẾT QUẢ KIỂM ĐỊNH:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4e}")
    
    alpha = 0.05
    if p_val < alpha:
        print("=> KẾT LUẬN: BÁC BỎ H0. Có sự khác biệt ý nghĩa thống kê về áp suất giữa hai nhóm.")
    else:
        print("=> KẾT LUẬN: CHẤP NHẬN H0. Không có sự khác biệt ý nghĩa thống kê.")