import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd
import missingno as msno
import numpy as np
from scipy import stats
from scipy.stats import norm

def plot_missing_patterns(df, sample_size=1000, random_state=42):
    """
    Visualize missing data patterns using Missingno.

    This function provides:
    1. Bar chart: Missing value count by feature
    2. Matrix plot: Locality and structure of missingness
    3. Heatmap: Correlation of missingness between features
    """

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Missing value count by feature
    ax1 = fig.add_subplot(2, 2, 1)
    msno.bar(df, color="steelblue", fontsize=10, ax=ax1, sort='descending')
    ax1.set_title("Missing Values Count by Feature", fontsize=14, fontweight='bold', pad=20)

    # Plot 2: Missingness matrix (locality & structure)
    ax2 = fig.add_subplot(2, 2, 2)
    msno.matrix(
        df.sample(min(sample_size, len(df)), random_state=random_state),
        color=(0.2, 0.2, 0.2),
        sparkline=False,
        fontsize=10,
        ax=ax2
    )
    ax2.set_title("Missingness Matrix (Random 1000 samples)", fontsize=14, fontweight='bold', pad=20)

    # Plot 3: Correlation of missingness
    ax3 = fig.add_subplot(2, 1, 2)
    msno.heatmap(df, cmap='RdBu', fontsize=10, ax=ax3)
    ax3.set_title("Nullity Correlation: Do Features Go Missing Together?", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()


def plot_missing_analysis(df_weather):
    """Generate a 3-plot missing data analysis panel."""

    if not pd.api.types.is_datetime64_any_dtype(df_weather['Date']):
        df_weather['Date'] = pd.to_datetime(df_weather['Date'], errors='coerce')

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

    # Plot A – Missing over Time
    monthly_missing = df_weather.set_index('Date').resample('M').apply(
        lambda x: x.isna().mean().mean() * 100
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(monthly_missing.index, monthly_missing.values, linewidth=2)
    ax1.set_title('Global Missing Rate Over Time (Monthly)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg Missing Rate (%)')
    ax1.grid(True, alpha=0.3)

    # Plot B – Missingness by Location
    sensitive_cols = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']
    loc_missing = df_weather.groupby('Location')[sensitive_cols].apply(
        lambda x: x.isna().mean().mean() * 100
    ).sort_values(ascending=False).head(15)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(
        x=loc_missing.values,
        y=loc_missing.index,
        hue=loc_missing.index,
        palette='viridis',
        legend=False,
        ax=ax2
    )
    ax2.set_title('Top Locations with Highest Missingness', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Missing Rate (%)')

    # Plot C – Mechanism Test (Observed vs Missing)
    feature = 'Evaporation'
    obs = df_weather.loc[df_weather[feature].notna(), 'Rainfall']
    mis = df_weather.loc[df_weather[feature].isna(), 'Rainfall']
    ax3 = fig.add_subplot(gs[1, :])
    sns.kdeplot(obs, ax=ax3, label='Observed', fill=True, alpha=0.25)
    sns.kdeplot(mis, ax=ax3, label='Missing', fill=True, alpha=0.25)
    ax3.set_xlim(0, 20)
    ax3.set_title(f'Rainfall Distribution when {feature} is Missing vs Observed', fontsize=12, fontweight='bold')
    ax3.legend()
    plt.tight_layout()
    plt.show()

    t_stat, p_val = stats.ttest_ind(obs.dropna(), mis.dropna(), equal_var=False)

    return {
        "feature": feature,
        "obs_mean": obs.mean(),
        "mis_mean": mis.mean(),
        "p_value": p_val
    }


def analyze_numerical_feature(df: pd.DataFrame, column: str, figsize=(16, 6)) -> None:
    """
    Perform a comprehensive exploratory analysis for a single numerical feature.

    The analysis includes:
    - Descriptive statistics
    - Distribution shape diagnostics
    - Multiple visualizations (Histogram, Boxplot, Q-Q Plot, Density comparison)
    - Outlier detection using the IQR method
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    series = df[column]

    print(f"\nDETAILED NUMERICAL ANALYSIS: {column}")

    # DESCRIPTIVE STATISTICS
    stats_dict = {
        "Count": series.count(),
        "Mean": series.mean(),
        "Median": series.median(),
        "Std": series.std(),
        "Min": series.min(),
        "Q1": series.quantile(0.25),
        "Q3": series.quantile(0.75),
        "Max": series.max(),
        "IQR": series.quantile(0.75) - series.quantile(0.25),
        "Skewness": series.skew(),
        "Kurtosis": series.kurtosis(),
        "Missing_Rate_%": series.isnull().mean() * 100,
    }
    stats_df = (
        pd.DataFrame(stats_dict, index=["Value"])
        .T.round(4)
    )
    print("\DESCRIPTIVE STATISTICS")
    display(stats_df)

    skew = stats_dict["Skewness"]
    mean = stats_dict["Mean"]
    median = stats_dict["Median"]

    # VISUALIZATION PANEL
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4)
    ax_hist = fig.add_subplot(gs[0, :2])
    ax_box = fig.add_subplot(gs[0, 2:])
    ax_qq = fig.add_subplot(gs[1, :2])
    ax_density = fig.add_subplot(gs[1, 2:])

    # Histogram + KDE
    sns.histplot(series.dropna(), bins=50, kde=True, ax=ax_hist)
    ax_hist.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean:.2f}")
    ax_hist.axvline(median, color="green", linestyle="-", linewidth=2, label=f"Median = {median:.2f}")
    ax_hist.set_title(f"Histogram & KDE – {column}", fontweight="bold")
    ax_hist.set_xlabel(column)
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    # Boxplot
    sns.boxplot(x=series.dropna(), ax=ax_box)
    ax_box.set_title(f"Boxplot – {column}", fontweight="bold")
    ax_box.set_xlabel(column)

    # Q-Q Plot
    stats.probplot(series.dropna(), dist="norm", plot=ax_qq)
    ax_qq.set_title("Q-Q Plot (Normality Check)", fontweight="bold")
    ax_qq.grid(alpha=0.3)

    # Density comparison with fitted normal distribution
    if series.dropna().shape[0] > 0:
        mu, sigma = norm.fit(series.dropna())
        x = np.linspace(series.min(), series.max(), 200)
        ax_density.plot(x, norm.pdf(x, mu, sigma), linewidth=2, label="Fitted Normal")
        sns.kdeplot(series.dropna(), ax=ax_density, label="Empirical KDE")
        ax_density.set_title("Empirical vs Normal Distribution", fontweight="bold")
        ax_density.legend()
        ax_density.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # OUTLIER ANALYSIS (IQR METHOD)
    print("OUTLIER ANALYSIS (IQR METHOD)")
    Q1, Q3, IQR = stats_dict["Q1"], stats_dict["Q3"], stats_dict["IQR"]
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_rate = (outliers.count() / series.count() * 100) if series.count() > 0 else 0
    outlier_summary = (
        pd.DataFrame(
            {
                "Lower_Bound": lower_bound,
                "Q1": Q1,
                "Median": median,
                "Q3": Q3,
                "Upper_Bound": upper_bound,
                "Outlier_Count": outliers.count(),
                "Outlier_Rate_%": outlier_rate,
                "Observed_Min": stats_dict["Min"],
                "Observed_Max": stats_dict["Max"],
            },
            index=["Value"],
        )
        .T.round(4)
    )
    display(outlier_summary)

    if outliers.count() > 0:
        print(
            f"Outliers detected: {outliers.count()} observations "
            f"({outlier_rate:.2f}% of non-missing values)."
        )
        print(f" • Smallest outlier value: {outliers.min():.4f}")
        print(f" • Largest outlier value: {outliers.max():.4f}")
    else:   
        print("No significant outliers detected using the IQR criterion.")

def analyze_feature_group(df, group_name, features):
    print(f"FEATURE GROUP ANALYSIS: {group_name.upper()}")
    for feature in features:
        if feature in df.columns:
            analyze_numerical_feature(df, feature)
        else:
            print(f"Feature '{feature}' not found.")

