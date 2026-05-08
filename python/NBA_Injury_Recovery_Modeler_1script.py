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

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter, CoxPHFitter

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Creating a random number generator object for reproducibility
RNG = np.random.default_rng(seed=42)

# Plot style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

print('Imports loaded. Ready to build the dataset.')


# ===========================================================================
# Step 1 — Building the Injury Event Dataset
#
# The raw injury-report data is a daily stream of player-status snapshots.
# We flip this into an event log — one row per injury — with when the player
# went out, when they returned, and their pre/post performance.
# ===========================================================================

import time
import sqlite3
import os
import ast
from datetime import datetime, timedelta
from nbainjuries import injury
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players

CACHE = sqlite3.connect('nba_cache.db')


def fetch_injury_reports(seasons=('2023-24', '2024-25')):
    """
    Fetch injury reports by generating timestamps for each season.
    Reports are timestamped hourly at 5:30 PM ET.
    """
    frames = []

    # Season start dates (adjusted to actual NBA season openers)
    season_dates = {
        '2023-24': (datetime(2023, 10, 24), datetime(2024, 6, 30)),
        '2024-25': (datetime(2024, 10, 22), datetime(2025, 6, 30)),
    }

    for season in seasons:
        start_date, end_date = season_dates[season]
        current = start_date.replace(hour=17, minute=30)

        while current <= end_date:
            if injury.check_reportvalid(current):
                try:
                    report = injury.get_reportdata(current, return_df=True)
                    if report is not None and len(report) > 0:
                        report['season'] = season
                        report['report_date'] = current
                        frames.append(report)
                        print(f"Fetched {season} report for {current.date()}: {len(report)} entries")
                except Exception as e:
                    print(f"Failed {season} {current.date()}: {e}")

            time.sleep(0.5)  # Rate limiting
            current += timedelta(days=1)

        print(f"Completed season {season}\n")

    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()


def collapse_to_events(df):
    """
    Turn daily snapshots into one row per OUT-streak.
    Column names from nbainjuries: 'Player Name', 'Game Date',
    'Current Status', 'Reason', 'Team'
    """
    df['Game Date'] = pd.to_datetime(df['Game Date'])
    df = df.sort_values(['Player Name', 'Game Date'])

    # Mark OUT status
    df['is_out'] = (df['Current Status'] == 'Out').astype(int)

    # Create streak IDs for consecutive OUT periods
    df['streak_id'] = (df['is_out'].diff() == 1).groupby(df['Player Name']).cumsum()

    # Aggregate OUT streaks
    out_streaks = df[df['is_out'] == 1].groupby(['Player Name', 'streak_id']).agg(
        out_date=('Game Date', 'min'),
        last_out_date=('Game Date', 'max'),
        reason=('Reason', lambda s: s.mode().iat[0] if len(s.mode()) > 0 else s.iloc[0]),
        team=('Team', 'first'),
        season=('season', 'first'),
    ).reset_index()

    # Filter out non-injuries: rest, load management, G-League assignments, etc.
    mask = ~out_streaks['reason'].str.lower().str.contains(
        'rest|load|personal|not injury|g league|two-way',
        na=False,
        regex=True
    )

    return out_streaks[mask].copy()


def get_player_id(player_name):
    """
    Look up NBA player ID from name.
    Handles both "First Last" and "Last, First" formats.
    """
    all_players = players.get_players()

    # Reverse "Last, First" to "First Last"
    if ',' in player_name:
        parts = player_name.split(',')
        if len(parts) == 2:
            player_name = f"{parts[1].strip()} {parts[0].strip()}"

    # Exact match
    matches = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
    if matches:
        return matches[0]['id']

    # Partial match
    matches = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
    if matches:
        return matches[0]['id']

    return None


def get_player_birthdate(player_id, max_retries=3):
    """Fetch player birth date with retry logic."""
    for attempt in range(max_retries):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            birthdate_str = info['BIRTHDATE'].iloc[0]

            if pd.notna(birthdate_str):
                birthdate = pd.to_datetime(birthdate_str.split('T')[0])
                return birthdate
            return None

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
            else:
                return None
    return None


def calculate_age(birthdate, injury_date):
    """Calculate age in years at the time of injury."""
    if pd.isna(birthdate) or pd.isna(injury_date):
        return None

    birthdate = pd.to_datetime(birthdate)
    injury_date = pd.to_datetime(injury_date)

    age = injury_date.year - birthdate.year
    if (injury_date.month, injury_date.day) < (birthdate.month, birthdate.day):
        age -= 1

    return age


def attach_game_logs(events, save_every=10, output_file='nba_injury_analysis.csv'):
    """
    For each injury event, fetch game logs and compute pre/post performance.
    Critical fix: Fetches game logs for the correct season.
    Saves progress incrementally and can resume from where it left off.
    """
    enriched = []

    # Check for existing progress
    already_processed = set()
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        already_processed = set(existing_df['player_name'] + '_' + existing_df['out_date'].astype(str))
        enriched = existing_df.to_dict('records')
        print(f"Found existing file with {len(enriched)} events. Resuming...\n")

    print(f"Processing {len(events)} injury events ({len(already_processed)} already done)...\n")

    for idx, ev in events.iterrows():
        event_key = f"{ev['Player Name']}_{ev['out_date']}"

        # Skip if already processed
        if event_key in already_processed:
            continue

        print(f"[{idx+1}/{len(events)}] Processing {ev['Player Name']}...")

        player_id = get_player_id(ev['Player Name'])
        if player_id is None:
            print(f"Skipping - player not found")
            continue

        # CRITICAL: Get the injury season
        injury_season = ev.get('season', '2023-24')

        # Fetch game logs for the CORRECT SEASON
        try:
            logs = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=injury_season
            ).get_data_frames()[0]
            time.sleep(0.8)
        except Exception as e:
            print(f"Failed to fetch logs: {str(e)[:100]}")
            continue

        try:
            logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])

            # Games before injury
            pre = logs[logs['GAME_DATE'] < ev.out_date].head(10)
            # Games after return
            post = logs[logs['GAME_DATE'] > ev.last_out_date].head(20)

            if len(pre) >= 5 and len(post) >= 5:
                enriched.append({
                    'player_name': ev['Player Name'],
                    'team': ev['team'],
                    'season': injury_season,
                    'injury_reason': ev['reason'],
                    'out_date': ev['out_date'],
                    'return_date': ev['last_out_date'],
                    'days_out': (ev['last_out_date'] - ev['out_date']).days,

                    # Pre-injury stats
                    'pre_avg_pts': pre['PTS'].mean(),
                    'pre_avg_reb': pre['REB'].mean(),
                    'pre_avg_ast': pre['AST'].mean(),
                    'pre_avg_min': pre['MIN'].mean(),

                    # Post-injury stats
                    'post_pts': post['PTS'].tolist(),
                    'post_reb': post['REB'].tolist(),
                    'post_ast': post['AST'].tolist(),
                    'post_min': post['MIN'].tolist(),

                    'post_avg_pts': post['PTS'].mean(),
                    'post_avg_reb': post['REB'].mean(),
                    'post_avg_ast': post['AST'].mean(),
                    'post_avg_min': post['MIN'].mean(),
                })
                print(f"Success: {len(pre)} pre, {len(post)} post games")
            else:
                print(f"Skipping - insufficient data (pre: {len(pre)}, post: {len(post)})")

        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            continue

        # Save progress
        if len(enriched) > 0 and (len(enriched) - len(already_processed)) % save_every == 0:
            df_temp = pd.DataFrame(enriched)
            df_temp.to_csv(output_file, index=False)
            print(f"Saved progress: {len(enriched)} events total\n")

    # Final save
    final_df = pd.DataFrame(enriched)
    if len(final_df) > 0:
        final_df.to_csv(output_file, index=False)
        print(f"\n✅ Final save: {len(final_df)} injury recovery events")

    return final_df


def add_real_ages(df, cache_file='player_birthdates_cache.csv'):
    """
    Add real player ages to the dataframe.
    Uses a cache file to avoid re-fetching birth dates.
    """
    # Try to load cached birth dates
    try:
        birthdate_cache = pd.read_csv(cache_file)
        birthdate_cache['birthdate'] = pd.to_datetime(birthdate_cache['birthdate'])
        print(f"📂 Loaded {len(birthdate_cache)} cached birth dates")
    except FileNotFoundError:
        birthdate_cache = pd.DataFrame(columns=['player_name', 'player_id', 'birthdate'])
        print("No cache found, will fetch all birth dates")

    unique_players = df['player_name'].unique()
    print(f"\nFetching birth dates for {len(unique_players)} unique players...")

    new_birthdates = []

    for idx, player_name in enumerate(unique_players):
        # Check cache
        cached = birthdate_cache[birthdate_cache['player_name'] == player_name]
        if len(cached) > 0:
            print(f"[{idx+1}/{len(unique_players)}] {player_name} - cached")
            continue

        print(f"[{idx+1}/{len(unique_players)}] Fetching {player_name}...")

        player_id = get_player_id(player_name)
        if player_id is None:
            print(f"Player ID not found")
            new_birthdates.append({'player_name': player_name, 'player_id': None, 'birthdate': None})
            continue

        birthdate = get_player_birthdate(player_id)
        if birthdate:
            print(f"Born: {birthdate.date()}")
            new_birthdates.append({'player_name': player_name, 'player_id': player_id, 'birthdate': birthdate})
        else:
            print(f"Birth date not available")
            new_birthdates.append({'player_name': player_name, 'player_id': player_id, 'birthdate': None})

        time.sleep(0.8)

        # Save progress every 10 players
        if len(new_birthdates) > 0 and len(new_birthdates) % 10 == 0:
            temp_cache = pd.concat([birthdate_cache, pd.DataFrame(new_birthdates)], ignore_index=True)
            temp_cache.to_csv(cache_file, index=False)
            print(f"Saved progress\n")

    # Combine cache with new data
    if new_birthdates:
        birthdate_cache = pd.concat([birthdate_cache, pd.DataFrame(new_birthdates)], ignore_index=True)
        birthdate_cache.to_csv(cache_file, index=False)
        print(f"\n✅ Saved complete birth date cache: {len(birthdate_cache)} players")

    # Merge with injury data
    df = df.merge(birthdate_cache[['player_name', 'birthdate']], on='player_name', how='left')

    # Calculate age at injury
    df['age'] = df.apply(lambda row: calculate_age(row['birthdate'], row['out_date']), axis=1)

    print(f"\n Age statistics:")
    print(f"  - Events with age data: {df['age'].notna().sum()} / {len(df)}")
    print(f"  - Age range: {df['age'].min():.0f} - {df['age'].max():.0f}")
    print(f"  - Mean age: {df['age'].mean():.1f}")

    df = df.drop('birthdate', axis=1)
    return df


def extract_injury_type(reason):
    """Extract injury type from free-text reason."""
    if pd.isna(reason):
        return 'Unknown'

    reason_lower = reason.lower()

    # Order matters - check specific before general
    if 'knee' in reason_lower:
        return 'Knee'
    elif 'ankle' in reason_lower:
        return 'Ankle'
    elif 'hamstring' in reason_lower:
        return 'Hamstring'
    elif 'back' in reason_lower or 'spine' in reason_lower:
        return 'Back'
    elif 'shoulder' in reason_lower:
        return 'Shoulder'
    elif 'hip' in reason_lower:
        return 'Hip'
    elif 'foot' in reason_lower or 'toe' in reason_lower:
        return 'Foot'
    elif 'hand' in reason_lower or 'finger' in reason_lower or 'wrist' in reason_lower:
        return 'Hand/Wrist'
    elif 'calf' in reason_lower:
        return 'Calf'
    elif 'groin' in reason_lower:
        return 'Groin'
    elif 'quad' in reason_lower:
        return 'Quadriceps'
    elif 'illness' in reason_lower or 'flu' in reason_lower or 'covid' in reason_lower:
        return 'Illness'
    elif 'concussion' in reason_lower or 'head' in reason_lower:
        return 'Concussion'
    else:
        return 'Other'


def extract_laterality(text):
    """Extract left/right/bilateral from injury text."""
    if not isinstance(text, str):
        return None
    t = text.lower()
    if 'bilateral' in t: return 'bilateral'
    if 'left' in t: return 'left'
    if 'right' in t: return 'right'
    return None


# ---------------------------------------------------------------------------
# Run the data pipeline
# ---------------------------------------------------------------------------

# Step 1: Fetch injury reports
injury_df = fetch_injury_reports(seasons=('2023-24', '2024-25'))
print(f"\nFetched {len(injury_df)} injury report entries")

# Step 2: Collapse to injury events
events_df = collapse_to_events(injury_df)
print(f"Found {len(events_df)} injury events to process\n")

# Step 3: Attach game logs (this takes a while - saves progress incrementally)
final_df = attach_game_logs(
    events_df,
    save_every=5,
    output_file='nba_injury_analysis_v2.csv'
)
print(f"\n✓ Complete! {len(final_df)} events exported")

# Step 4: Add real player ages (cached)
final_df = add_real_ages(final_df)

# Step 5: Add injury classification + games missed, save complete dataset
final_df['injury_type'] = final_df['injury_reason'].apply(extract_injury_type)
final_df['laterality'] = final_df['injury_reason'].apply(extract_laterality)
final_df['games_missed'] = (final_df['days_out'] / 2).round().astype(int)

final_df.to_csv('nba_injury_analysis_complete.csv', index=False)
print(f"✓ Saved complete dataset: {len(final_df)} events")
print(f"\nInjury type distribution:")
print(final_df['injury_type'].value_counts())


# ===========================================================================
# Load the complete dataset
#
# If you've already run the pipeline above, you can comment out steps 1-5
# and start here by loading the saved CSV.
# ===========================================================================

events_df = pd.read_csv('nba_injury_analysis_complete.csv')

# Parse list columns from string representation back to Python lists
list_cols = ['post_pts', 'post_reb', 'post_ast', 'post_min']
for col in list_cols:
    events_df[col] = events_df[col].apply(ast.literal_eval)

print(f"Loaded {len(events_df)} injury events")
print(f"Columns: {events_df.columns.tolist()}")
print(events_df.head())


# ===========================================================================
# Dataset Overview: injury counts and games missed
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

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
plt.show()

print('\nSummary by injury type:')
print(events_df.groupby('injury_type').agg(
    n=('games_missed', 'size'),
    median_games_missed=('games_missed', 'median'),
    mean_age=('age', 'mean'),
).round(1))


# ===========================================================================
# Step 2 — NLP Injury Classification
#
# The injury_reason column holds messy free text. A keyword-based classifier
# handles 95%+ of cases. We validate accuracy and inspect edge cases.
# ===========================================================================

# Show sample of classified injuries with laterality
sample = events_df[['injury_reason', 'injury_type', 'laterality']].sample(12, random_state=1)
print("Sample of classified injuries:")
print(sample.to_string(index=False))

# Distribution of unclassified (Other)
other_reasons = events_df[events_df['injury_type'] == 'Other']['injury_reason'].value_counts()
print(f"\nInjury reasons classified as 'Other' ({len(other_reasons)} unique):")
print(other_reasons.head(15))


# ===========================================================================
# Step 3 — Performance Delta Analysis
#
# For each injury event we compare post-return performance to the pre-injury
# baseline across three windows:
#   - Games 1-5:  first impression
#   - Games 6-10: early recovery
#   - Games 11-20: settled-in
#
# For each (injury_type, window, metric) cell we run a paired t-test
# (H0: mean delta = 0) and compute Cohen's d as the effect size.
# Cohen's d rules of thumb: 0.2 = small, 0.5 = medium, 0.8 = large.
# ===========================================================================

WINDOWS = {
    '1-5':   (0, 5),
    '6-10':  (5, 10),
    '11-20': (10, 20),
}
METRICS = ['pts', 'reb', 'ast', 'min']  # Matches our actual column names


def compute_window_delta(row, metric, window):
    """Post-window mean minus pre-injury baseline."""
    start, end = WINDOWS[window]
    post_vals = row[f'post_{metric}'][start:end]
    if len(post_vals) == 0:
        return np.nan
    return float(np.mean(post_vals)) - float(row[f'pre_avg_{metric}'])


# Build long-form delta table: one row per (event × metric × window)
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
print(f"Delta table: {len(deltas)} rows")
print(deltas.head(8))


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

print(f'Total tests: {n_tests}, Bonferroni alpha = 0.05/{n_tests} = {0.05/n_tests:.2e}')
print(f'Significant after correction: {int(summary["sig"].sum())}')


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

print('Mean PTS delta (post − pre) by injury type and window:')
print(pts_summary.round(2))
print()
print("Cohen's d (effect size) for PTS delta:")
print(pts_d.round(2))


# --- Heatmaps: PTS delta and Cohen's d ---

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(pts_summary, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=axes[0], cbar_kws={'label': 'mean PTS delta'})
axes[0].set_title('Mean PTS drop by injury type & window')
axes[0].set_ylabel('injury type')

sns.heatmap(pts_d, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=axes[1], cbar_kws={'label': "Cohen's d"})
axes[1].set_title("Cohen's d (effect size) for PTS delta")
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# --- All metrics heatmap for window 1-5 ---

plt.figure(figsize=(11, 5))
g = (summary.query("window == '1-5'")
     .pivot(index='injury_type', columns='metric', values='cohens_d')
     [METRICS])
sns.heatmap(g, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            cbar_kws={'label': "Cohen's d"})
plt.title("Cohen's d across all metrics — games 1–5 after return")
plt.ylabel('injury type')
plt.tight_layout()
plt.show()


# ===========================================================================
# Step 4 — Survival Analysis: Time-to-Full-Recovery
#
# The delta analysis tells us HOW BIG the drop is.
# Survival analysis tells us HOW LONG it takes to disappear.
#
# We define recovery per event: the first game where the player's rolling
# 5-game PTS average is no longer meaningfully below their pre-injury
# baseline (within 0.25σ). Events that never reach this threshold within
# 20 games are right-censored.
# ===========================================================================

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

print(f'Recovery events: {len(recovery)}')
print(f'Recovered within 20 games: {int(recovery["recovered"].sum())} '
      f'({recovery["recovered"].mean():.1%})')
print(f'Right-censored: {int((~recovery["recovered"]).sum())}')
print('\nRecovery rate by injury type:')
print(recovery.groupby('injury_type').agg(
    n=('recovered', 'size'),
    recovery_rate=('recovered', 'mean'),
    median_games=('games_to_recovery', 'median'),
).round(2))


# --- Kaplan-Meier survival curves by injury type ---

fig, ax = plt.subplots(figsize=(11, 6))

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
    kmf.plot_survival_function(ax=ax, ci_show=False, color=color)

ax.set_title('Kaplan-Meier Survival Analysis: Probability of NOT YET fully recovered, by injury type')
ax.set_xlabel('Games since return')
ax.set_ylabel('P(still below baseline)')
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()


# --- Cox Proportional Hazards regression ---

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
cph.print_summary(decimals=3)


# --- Verify proportional hazards assumption ---

ph_test = cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)


# ===========================================================================
# Step 5 — DTW Clustering of Recovery Trajectories
#
# The delta analysis averages across players. Survival analysis estimates a
# single curve per injury type. Both miss the fact that recovery shapes vary
# even within an injury category:
#
#   - Fast recoverers — bounce back within a few games.
#   - Slow recoverers — creep back up gradually over many weeks.
#   - Performance dippers — drop further before climbing.
#   - Bouncers — overshoot baseline, possibly from a rest effect.
#
# Dynamic Time Warping (DTW) measures trajectory shape similarity regardless
# of time alignment, making it ideal for clustering recovery curves.
# ===========================================================================

TRAJECTORY_METRICS = ['pts', 'min']  # Metrics with list data available
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
        cols.append(traj / baseline)  # Fraction of baseline
    if valid:
        trajectories.append(np.column_stack(cols))
        event_index.append(idx)

X_raw = np.array(trajectories)
print(f'Trajectory tensor shape: {X_raw.shape}  (events, games, metrics)')

scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
X = scaler.fit_transform(X_raw)
print(f'Normalized tensor shape: {X.shape}')


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
print('Cluster sizes:')
print(cluster_sizes)

events_clustered = events_df.iloc[event_index].copy()
events_clustered['cluster'] = labels


# --- Plot cluster centroids ---

fig, axes = plt.subplots(1, len(TRAJECTORY_METRICS),
                          figsize=(5 * len(TRAJECTORY_METRICS), 4.5),
                          sharey=False)

if len(TRAJECTORY_METRICS) == 1:
    axes = [axes]  # Handle single metric case

cluster_colors = sns.color_palette('Set2', n_colors=N_CLUSTERS)

cluster_names = {
    0: 'Fast Recoverers',
    1: 'Slow Recoverers',
    2: 'Mid-Return Dippers',
    3: 'Bouncers',
}
for ax, m_idx, metric in zip(axes, range(len(TRAJECTORY_METRICS)), TRAJECTORY_METRICS):
    for c in range(N_CLUSTERS):
        cluster_mask = labels == c
        mean_traj = X_raw[cluster_mask, :, m_idx].mean(axis=0)
        ax.plot(np.arange(1, N_GAMES + 1), mean_traj,
                color=cluster_colors[c], linewidth=2.2,
                label=f'cluster {c} (n={int(cluster_mask.sum())})')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_title(f'{metric} (fraction of pre-injury baseline)')
    ax.set_xlabel('games since return')
    ax.set_ylabel('value / baseline')
    if m_idx == 0:
        ax.legend(fontsize=9, loc='upper right')

plt.suptitle('Mean recovery trajectory per cluster', y=1.02, fontsize=13)
plt.tight_layout()
plt.show()


# --- Cluster diagnostic: early vs late performance ---

for c in range(N_CLUSTERS):
    cluster_mask = labels == c
    mean_early = X_raw[cluster_mask, :5, 0].mean()   # Games 1-5, pts
    mean_late = X_raw[cluster_mask, 15:, 0].mean()    # Games 16-20, pts
    print(f"Cluster {c} (n={int(cluster_mask.sum())}): "
          f"early avg={mean_early:.2f}, late avg={mean_late:.2f}")


# --- Cluster profiles ---

profile = events_clustered.groupby('cluster').agg(
    n=('cluster', 'size'),
    mean_age=('age', 'mean'),
    mean_games_missed=('games_missed', 'mean'),
    pct_knee=('injury_type', lambda s: (s == 'Knee').mean()),
    pct_foot=('injury_type', lambda s: (s == 'Foot').mean()),
    pct_ankle=('injury_type', lambda s: (s == 'Ankle').mean()),
    pct_concussion=('injury_type', lambda s: (s == 'Concussion').mean()),
).round(2)
print('Cluster profiles:')
print(profile)


# --- Injury-type composition heatmap within each cluster ---

mix = pd.crosstab(events_clustered['cluster'], events_clustered['injury_type'],
                   normalize='index')
plt.figure(figsize=(10, 4))
sns.heatmap(mix, annot=True, fmt='.2f', cmap='Blues',
            cbar_kws={'label': 'share of cluster'})
plt.title('Injury-type composition within each cluster')
plt.ylabel('cluster')
plt.xlabel('injury type')
plt.tight_layout()
plt.show()
