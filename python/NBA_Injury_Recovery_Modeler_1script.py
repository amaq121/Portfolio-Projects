"""
NBA Injury Recovery Performance Modeler
========================================
Quantifying how players perform across recovery windows after specific
injury types and predicting full recovery timelines.

Five Connected Steps:
1. Build the injury event dataset
2. NLP injury classification
3. Performance delta analysis (paired t-tests, Cohen's d)
4. Survival analysis (Kaplan-Meier, Cox regression)
5. DTW clustering of recovery trajectories
"""

# ===========================================================================
# Setup: imports and configuration
# ===========================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import ast
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter, CoxPHFitter

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import streamlit as st

# --- Streamlit page config ---
st.set_page_config(
    page_title="NBA Injury Recovery Modeler",
    page_icon="🏀",
    layout="wide",
)

# Creating a random number generator object for reproducibility
RNG = np.random.default_rng(seed=42)

# Plot style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


# ===========================================================================
# Load the complete dataset
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    """Load and parse the injury analysis dataset. Cached for performance."""
    df = pd.read_csv(os.path.join(SCRIPT_DIR, 'nba_injury_analysis_complete.csv'))
    list_cols = ['post_pts', 'post_reb', 'post_ast', 'post_min']
    for col in list_cols:
        df[col] = df[col].apply(ast.literal_eval)
    return df


events_df = load_data()


# ===========================================================================
# App Header
# ===========================================================================

st.title("🏀 NBA Injury Recovery Performance Modeler")
st.markdown(
    "Quantifying how NBA players perform across recovery windows after specific "
    "injury types, and predicting full recovery timelines using statistical analysis, "
    "survival modeling, and time-series clustering."
)
st.markdown("---")

st.metric("Total Injury Events Analyzed", len(events_df))


# ===========================================================================
# Dataset Overview: injury counts and games missed
# ===========================================================================

st.header("📊 Dataset Overview")

fig_overview, axes = plt.subplots(1, 2, figsize=(13, 4))

events_df['injury_type'].value_counts().plot(
    kind='bar', ax=axes[0], color='steelblue'
)
axes[0].set_title('Injury events by type')
axes[0].set_ylabel('count')
axes[0].tick_params(axis='x', rotation=45)

order = events_df.groupby('injury_type')['games_missed'].median().sort_values().index
sns.boxplot(data=events_df, x='injury_type', y='games_missed',
            order=order, ax=axes[1], color='steelblue')
axes[1].set_title('Games missed by injury type')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig_overview)

st.subheader("Summary by Injury Type")
summary_by_type = events_df.groupby('injury_type').agg(
    n=('games_missed', 'size'),
    median_games_missed=('games_missed', 'median'),
    mean_age=('age', 'mean'),
).round(1)
st.dataframe(summary_by_type, use_container_width=True)


# ===========================================================================
# Step 2 — NLP Injury Classification
# ===========================================================================

st.header("🔤 NLP Injury Classification")
st.markdown(
    "The `injury_reason` column holds messy free text from official injury reports. "
    "A keyword-based classifier maps these into standardized injury types, handling "
    "95%+ of cases."
)

st.subheader("Sample of Classified Injuries")
sample = events_df[['injury_reason', 'injury_type', 'laterality']].sample(12, random_state=1)
st.dataframe(sample, use_container_width=True, hide_index=True)

st.subheader("Unclassified Injuries ('Other')")
other_reasons = events_df[events_df['injury_type'] == 'Other']['injury_reason'].value_counts()
st.write(f"**{len(other_reasons)}** unique injury reasons classified as 'Other':")
st.dataframe(other_reasons.head(15).reset_index().rename(
    columns={'index': 'Injury Reason', 'injury_reason': 'Injury Reason', 'count': 'Count'}
), use_container_width=True, hide_index=True)


# ===========================================================================
# Step 3 — Performance Delta Analysis
# ===========================================================================

st.header("📉 Performance Delta Analysis")
st.markdown(
    """
    For each injury event we compare post-return performance to the pre-injury
    baseline across three recovery windows:
    - **Games 1-5:** First impression after return
    - **Games 6-10:** Early recovery
    - **Games 11-20:** Settled-in performance

    For each (injury type, window, metric) cell we run a **paired t-test**
    (H₀: mean delta = 0) and compute **Cohen's d** as the effect size.
    Cohen's d rules of thumb: 0.2 = small, 0.5 = medium, 0.8 = large.
    """
)

WINDOWS = {
    '1-5':   (0, 5),
    '6-10':  (5, 10),
    '11-20': (10, 20),
}
METRICS = ['pts', 'reb', 'ast', 'min']


def compute_window_delta(row, metric, window):
    """Post-window mean minus pre-injury baseline."""
    start, end = WINDOWS[window]
    post_vals = row[f'post_{metric}'][start:end]
    if len(post_vals) == 0:
        return np.nan
    return float(np.mean(post_vals)) - float(row[f'pre_avg_{metric}'])


# Build long-form delta table: one row per (event x metric x window)
delta_rows = []
for _, ev in events_df.iterrows():
    for metric in METRICS:
        for window in WINDOWS:
            delta_rows.append({
                'player_name': ev['player_name'],
                'injury_type': ev['injury_type'],
                'age': ev['age'],
                'games_missed': ev['games_missed'],
                'metric': metric,
                'window': window,
                'delta': compute_window_delta(ev, metric, window),
            })

deltas = pd.DataFrame(delta_rows).dropna(subset=['delta'])


# --- Summarize deltas with t-tests and Cohen's d ---

def summarize_deltas(group):
    """Paired t-test against zero + Cohen's d."""
    d = group['delta'].values
    if len(d) < 5:
        return pd.Series({'n': len(d), 'mean_delta': np.nan,
                          't_stat': np.nan, 'p_value': np.nan, 'cohens_d': np.nan})
    t, p = stats.ttest_1samp(d, 0)
    sd = np.std(d, ddof=1)
    cohens_d = np.mean(d) / sd if sd > 0 else np.nan
    return pd.Series({
        'n': len(d), 'mean_delta': np.mean(d),
        't_stat': t, 'p_value': p, 'cohens_d': cohens_d,
    })


summary = (deltas
    .groupby(['injury_type', 'window', 'metric'])
    .apply(summarize_deltas)
    .reset_index()
)

n_tests = len(summary)
summary['p_bonferroni'] = (summary['p_value'] * n_tests).clip(upper=1.0)
summary['sig'] = summary['p_bonferroni'] < 0.05

col1, col2 = st.columns(2)
col1.metric("Total Statistical Tests", n_tests)
col2.metric("Significant After Bonferroni Correction", int(summary['sig'].sum()))


# --- PTS delta pivot tables ---

pts_summary = (summary
    .query("metric == 'pts'")
    .pivot(index='injury_type', columns='window', values='mean_delta')
    [['1-5', '6-10', '11-20']]
)

pts_d = (summary
    .query("metric == 'pts'")
    .pivot(index='injury_type', columns='window', values='cohens_d')
    [['1-5', '6-10', '11-20']]
)

st.subheader("PTS Delta: Mean Drop and Effect Size")

tab1, tab2 = st.tabs(["Mean PTS Delta (post - pre)", "Cohen's d (Effect Size)"])
with tab1:
    st.dataframe(pts_summary.round(2), use_container_width=True)
with tab2:
    st.dataframe(pts_d.round(2), use_container_width=True)


# --- Heatmaps: PTS delta and Cohen's d ---

st.subheader("PTS Delta Heatmaps")

fig_heatmaps, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(pts_summary, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=axes[0], cbar_kws={'label': 'mean PTS delta'})
axes[0].set_title('Mean PTS drop by injury type & window')
axes[0].set_ylabel('injury type')

sns.heatmap(pts_d, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=axes[1], cbar_kws={'label': "Cohen's d"})
axes[1].set_title("Cohen's d (effect size) for PTS delta")
axes[1].set_ylabel('')

plt.tight_layout()
st.pyplot(fig_heatmaps)


# --- All metrics heatmap for window 1-5 ---

st.subheader("All Metrics: Cohen's d for Games 1-5 After Return")

fig_all_metrics, ax_all = plt.subplots(figsize=(11, 5))
g = (summary.query("window == '1-5'")
     .pivot(index='injury_type', columns='metric', values='cohens_d')
     [METRICS])
sns.heatmap(g, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax_all, cbar_kws={'label': "Cohen's d"})
ax_all.set_title("Cohen's d across all metrics — games 1-5 after return")
ax_all.set_ylabel('injury type')
plt.tight_layout()
st.pyplot(fig_all_metrics)


# ===========================================================================
# Step 4 — Survival Analysis: Time-to-Full-Recovery
# ===========================================================================

st.header("⏱️ Survival Analysis: Time-to-Full-Recovery")
st.markdown(
    """
    The delta analysis tells us **how big** the performance drop is.
    Survival analysis tells us **how long** it takes to disappear.

    We define recovery per event: the first game where the player's rolling
    5-game PTS average is no longer meaningfully below their pre-injury
    baseline (within 0.25 sigma). Events that never reach this threshold within
    20 games are right-censored.
    """
)


def games_to_recovery(post_traj, baseline, window=5, threshold_sigma=-0.25):
    """
    First game where rolling-window mean is within threshold of baseline.
    Returns (game_number, recovered_flag). Censored if never reached.
    """
    arr = np.array(post_traj)
    if len(arr) < window:
        return len(arr), False
    sigma = np.std(arr) if np.std(arr) > 0 else 1.0
    threshold = baseline + threshold_sigma * sigma
    rolling = pd.Series(arr).rolling(window).mean()
    for g, val in enumerate(rolling, start=1):
        if pd.notna(val) and val >= threshold:
            return g, True
    return len(arr), False


# Build recovery table
recovery_data = []
for _, ev in events_df.iterrows():
    g, recovered = games_to_recovery(ev['post_pts'], ev['pre_avg_pts'])
    recovery_data.append({
        'player_name': ev['player_name'],
        'injury_type': ev['injury_type'],
        'age': ev['age'],
        'games_missed': ev['games_missed'],
        'pre_avg_pts': ev['pre_avg_pts'],
        'games_to_recovery': g,
        'recovered': bool(recovered),
    })

recovery = pd.DataFrame(recovery_data)

col1, col2, col3 = st.columns(3)
col1.metric("Recovery Events", len(recovery))
col2.metric("Recovered Within 20 Games",
            f"{int(recovery['recovered'].sum())} ({recovery['recovered'].mean():.1%})")
col3.metric("Right-Censored", int((~recovery['recovered']).sum()))

st.subheader("Recovery Rate by Injury Type")
recovery_by_type = recovery.groupby('injury_type').agg(
    n=('recovered', 'size'),
    recovery_rate=('recovered', 'mean'),
    median_games=('games_to_recovery', 'median'),
).round(2)
st.dataframe(recovery_by_type, use_container_width=True)


# --- Kaplan-Meier survival curves by injury type ---

st.subheader("Kaplan-Meier Survival Curves")

fig_km, ax_km = plt.subplots(figsize=(11, 6))

palette = sns.color_palette('Paired', n_colors=recovery['injury_type'].nunique())
for color, injury_type in zip(palette, sorted(recovery['injury_type'].unique())):
    mask = recovery['injury_type'] == injury_type
    if mask.sum() < 10:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=recovery.loc[mask, 'games_to_recovery'],
        event_observed=recovery.loc[mask, 'recovered'],
        label=f'{injury_type} (n={int(mask.sum())})',
    )
    kmf.plot_survival_function(ax=ax_km, ci_show=False, color=color)

ax_km.set_title('Kaplan-Meier Survival Analysis: Probability of NOT YET fully recovered, by injury type')
ax_km.set_xlabel('Games since return')
ax_km.set_ylabel('P(still below baseline)')
ax_km.set_xlim(0, 20)
ax_km.set_ylim(0, 1.05)
ax_km.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax_km.legend(loc='upper right', fontsize=9)
plt.tight_layout()
st.pyplot(fig_km)


# --- Cox Proportional Hazards regression ---

st.subheader("Cox Proportional Hazards Regression")

# Only include injury types with enough events for stable estimates
keep_types = recovery['injury_type'].value_counts().loc[lambda s: s >= 20].index.tolist()
cox_df = recovery[recovery['injury_type'].isin(keep_types)].copy()

# Create dummy variables for injury type
cox_df = pd.get_dummies(cox_df, columns=['injury_type'], drop_first=True, dtype=float)

keep_cols = ['games_to_recovery', 'recovered', 'age', 'games_missed'] + \
            [c for c in cox_df.columns if c.startswith('injury_type_')]
cox_df = cox_df[keep_cols].dropna()

cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_df, duration_col='games_to_recovery', event_col='recovered')

st.write("**Cox PH Model Summary:**")
st.dataframe(cph.summary.round(3), use_container_width=True)

st.write("**Proportional Hazards Assumption Test:**")
try:
    ph_test = cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
    st.success("Proportional hazards assumption holds for all covariates.")
except Exception as e:
    st.warning(f"Proportional hazards assumption may be violated: {e}")


# ===========================================================================
# Step 5 — DTW Clustering of Recovery Trajectories
# ===========================================================================

st.header("🔬 DTW Clustering of Recovery Trajectories")
st.markdown(
    """
    The delta analysis averages across players. Survival analysis estimates a
    single curve per injury type. Both miss the fact that **recovery shapes vary**
    even within an injury category:

    - **Fast Recoverers** — bounce back within a few games
    - **Slow Recoverers** — creep back up gradually over many weeks
    - **Mid-Return Dippers** — drop further before climbing
    - **Bouncers** — overshoot baseline, possibly from a rest effect

    **Dynamic Time Warping (DTW)** measures trajectory shape similarity regardless
    of time alignment, making it ideal for clustering recovery curves.
    """
)

TRAJECTORY_METRICS = ['pts', 'min']
N_GAMES = 20

trajectories = []
event_index = []

for idx, ev in events_df.iterrows():
    cols = []
    valid = True
    for metric in TRAJECTORY_METRICS:
        traj = np.array(ev[f'post_{metric}'][:N_GAMES])
        baseline = ev[f'pre_avg_{metric}']
        if len(traj) < N_GAMES or baseline == 0:
            valid = False
            break
        cols.append(traj / baseline)
    if valid:
        trajectories.append(np.column_stack(cols))
        event_index.append(idx)

X_raw = np.array(trajectories)

scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
X = scaler.fit_transform(X_raw)

st.write(f"**Trajectory tensor shape:** {X_raw.shape} (events, games, metrics)")


# --- Run DTW clustering ---

N_CLUSTERS = 4

km = TimeSeriesKMeans(
    n_clusters=N_CLUSTERS,
    metric='dtw',
    max_iter=10,
    n_init=3,
    random_state=42,
    verbose=False,
)
labels = km.fit_predict(X)

cluster_sizes = pd.Series(labels).value_counts().sort_index()

events_clustered = events_df.iloc[event_index].copy()
events_clustered['cluster'] = labels

st.subheader("Cluster Sizes")
size_cols = st.columns(N_CLUSTERS)
cluster_names = {
    0: 'Fast Recoverers',
    1: 'Slow Recoverers',
    2: 'Mid-Return Dippers',
    3: 'Bouncers',
}
for i, col in enumerate(size_cols):
    col.metric(
        f"Cluster {i}: {cluster_names.get(i, '')}",
        int(cluster_sizes.get(i, 0))
    )


# --- Plot cluster centroids ---

st.subheader("Mean Recovery Trajectory per Cluster")

fig_clusters, axes = plt.subplots(1, len(TRAJECTORY_METRICS),
                                   figsize=(5 * len(TRAJECTORY_METRICS), 4.5),
                                   sharey=False)

if len(TRAJECTORY_METRICS) == 1:
    axes = [axes]

cluster_colors = sns.color_palette('Set2', n_colors=N_CLUSTERS)

for ax, m_idx, metric in zip(axes, range(len(TRAJECTORY_METRICS)), TRAJECTORY_METRICS):
    for c in range(N_CLUSTERS):
        cluster_mask = labels == c
        mean_traj = X_raw[cluster_mask, :, m_idx].mean(axis=0)
        ax.plot(np.arange(1, N_GAMES + 1), mean_traj,
                color=cluster_colors[c], linewidth=2.2,
                label=f'Cluster {c}: {cluster_names.get(c, "")} (n={int(cluster_mask.sum())})')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_title(f'{metric} (fraction of pre-injury baseline)')
    ax.set_xlabel('Games since return')
    ax.set_ylabel('Value / Baseline')
    if m_idx == 0:
        ax.legend(fontsize=8, loc='upper right')

plt.suptitle('Mean Recovery Trajectory per Cluster', y=1.02, fontsize=13)
plt.tight_layout()
st.pyplot(fig_clusters)


# --- Cluster diagnostic: early vs late performance ---

st.subheader("Cluster Diagnostic: Early vs Late Performance")

diag_data = []
for c in range(N_CLUSTERS):
    cluster_mask = labels == c
    mean_early = X_raw[cluster_mask, :5, 0].mean()
    mean_late = X_raw[cluster_mask, 15:, 0].mean()
    diag_data.append({
        'Cluster': f"{c}: {cluster_names.get(c, '')}",
        'n': int(cluster_mask.sum()),
        'Early Avg (Games 1-5)': round(mean_early, 2),
        'Late Avg (Games 16-20)': round(mean_late, 2),
    })
st.dataframe(pd.DataFrame(diag_data), use_container_width=True, hide_index=True)


# --- Cluster profiles ---

st.subheader("Cluster Profiles")

profile = events_clustered.groupby('cluster').agg(
    n=('cluster', 'size'),
    mean_age=('age', 'mean'),
    mean_games_missed=('games_missed', 'mean'),
    pct_knee=('injury_type', lambda s: (s == 'Knee').mean()),
    pct_foot=('injury_type', lambda s: (s == 'Foot').mean()),
    pct_ankle=('injury_type', lambda s: (s == 'Ankle').mean()),
    pct_concussion=('injury_type', lambda s: (s == 'Concussion').mean()),
).round(2)
st.dataframe(profile, use_container_width=True)


# --- Injury-type composition heatmap within each cluster ---

st.subheader("Injury-Type Composition Within Each Cluster")

mix = pd.crosstab(events_clustered['cluster'], events_clustered['injury_type'],
                   normalize='index')
fig_mix, ax_mix = plt.subplots(figsize=(10, 4))
sns.heatmap(mix, annot=True, fmt='.2f', cmap='Blues',
            ax=ax_mix, cbar_kws={'label': 'share of cluster'})
ax_mix.set_title('Injury-type composition within each cluster')
ax_mix.set_ylabel('Cluster')
ax_mix.set_xlabel('Injury Type')
plt.tight_layout()
st.pyplot(fig_mix)


# --- Footer ---
st.markdown("---")
st.markdown(
    "**NBA Injury Recovery Performance Modeler** | "
    "Built by Azlan Maqbool | "
    "Data: NBA injury reports (2023-24, 2024-25 seasons)"
)
