import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
import pandas as pd
import missingno as msno

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
