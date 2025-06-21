from typing import Optional
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio

def assign_labels_from_index(pii_series: pd.Series) -> pd.Series:
    """
    Assign qualitative tiers to players based on their Player Impact Index (PII) percentile.

    Tiers:
    - ≥ 90th: Superstar / MVP Candidate
    - ≥ 75th: All-Star Caliber
    - ≥ 55th: Starter / Solid Contributor
    - ≥ 35th: Role Player / Specialist
    - < 35th: Low Impact / Bench
    """
    q90 = pii_series.quantile(0.90)
    q75 = pii_series.quantile(0.75)
    q55 = pii_series.quantile(0.55)
    q35 = pii_series.quantile(0.35)

    def label(pii):
        if pii >= q90:
            return "Superstar / MVP Candidate"
        elif pii >= q75:
            return "All-Star Caliber"
        elif pii >= q55:
            return "Starter / Solid Contributor"
        elif pii >= q35:
            return "Role Player / Specialist"
        else:
            return "Low Impact / Bench"

    return pii_series.apply(label)

def compute_pii(df: pd.DataFrame, age_for_is_under: int = 25) -> pd.Series:
    """
    Compute Player Impact Index (PII) — a composite score based on rank-normalized metrics,
    adjusted for age (bonus for U25 players).

    Requires:
    - Rank-normalized metrics with 'rnk_' prefix.
    - Column 'AGE' for age-based bonuses.
    """
    df["Is_Under"] = (df["AGE"] <= age_for_is_under).astype(int)

    pii = (
        0.16 * df["rnk_PTS/G"] +
        0.10 * df["rnk_USG%"] +
        0.08 * df["rnk_AST/TO"] +
        0.08 * df["rnk_TS%"] +
        0.06 * df["rnk_eFG%"] +
        0.10 * df["rnk_PER"] +
        0.07 * df["rnk_NET RTG"] +
        0.06 * df["rnk_W%"] +
        0.04 * df["rnk_AST%"] -
        0.04 * df["rnk_TO Ratio"] -
        0.06 * df["rnk_DEF RTG"] +
        0.05 * df["rnk_TR%"] +
        0.04 * df["rnk_ST%"] +
        0.03 * df["rnk_BLK%"] +
        0.03 * df["rnk_Min/G"] +
        0.15 * df["Is_Under"] -
        0.20 * df["AGE"]
    ).round(4)


    return pii

def cluster_player_profiles(
    pkl_path: str,
    n_clusters: int = 5,
    min_games: int = 10,
    min_minutes: int = 5,
    show_plots: bool = True,
    return_data: bool = False,
    age_for_is_under: int = 25
) -> Optional[pd.DataFrame]:
    """
    Phase 1: Perform unsupervised clustering of NBA player seasons based on statistical profile.
    
    Steps:
    - Filters out insignificant contributions.
    - Normalizes performance metrics using percentile ranks.
    - Computes PII as a weighted index with age bias.
    - Applies KMeans clustering and PCA projection for visualization.
    - Assigns qualitative labels for interpretability.
    - Saves clustered data and generates visual plots.

    Returns:
    - DataFrame with cluster labels and PII if `return_data=True`.
    """
    print(f"[INFO] Loading dataset from: {pkl_path}")
    df = pd.read_pickle(pkl_path)

    # --- FILTERING: keep multi-season significant contributors ---
    df = df[(df['GP'] >= min_games) & (df['Min/G'] > min_minutes)]
    eligible = df.groupby('NAME')['SEASON'].nunique()
    df = df[df['NAME'].isin(eligible[eligible > 1].index)]

    # --- SELECT RELEVANT METRICS ---
    features = [
        'PTS/G', 'USG%', 'Min/G', 'TS%', 'eFG%', 'PER',
        'NET RTG', 'W%', 'AST/TO', 'AST%', 'TO Ratio',
        'DEF RTG', 'ST%', 'BLK%', 'TR%'
    ]
    df_clean = df.dropna(subset=features + ['AGE']).copy()
    df_clean[features] = df_clean[features].round(2)
    df_clean["AGE"] = df_clean["AGE"].astype(float)

    # --- NORMALIZE USING PERCENTILE RANKS ---
    for col in features:
        df_clean[f"rnk_{col}"] = df_clean[col].rank(pct=True)

    # --- COMPUTE PII ---
    df_clean["PII"] = compute_pii(df_clean, age_for_is_under)

    # --- KMEANS CLUSTERING ---
    X = df_clean[[f"rnk_{col}" for col in features]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df_clean["Cluster"] = kmeans.fit_predict(X)

    # --- PCA DIMENSIONALITY REDUCTION ---
    pcs = PCA(n_components=2).fit_transform(X)
    df_clean["PCA1"], df_clean["PCA2"] = pcs[:, 0], pcs[:, 1]

    # --- ASSIGN CLUSTER LABELS BASED ON PII ---
    df_clean["Cluster_Label"] = assign_labels_from_index(df_clean["PII"])

    # --- VISUALIZATION ---
    img_dir = "img"
    os.makedirs(img_dir, exist_ok=True)

    if show_plots:
        cluster_labels = sorted(df_clean["Cluster_Label"].unique())
        traces = []
        for label in cluster_labels:
            group = df_clean[df_clean["Cluster_Label"] == label]
            traces.append(go.Scatter(
                x=group["PCA1"],
                y=group["PCA2"],
                mode="markers",
                name=label,
                marker=dict(size=10, opacity=0.75, line=dict(width=1, color="black")),
                text=[f"<b>{r['NAME']}</b> ({r['SEASON']})<br>PTS/G: {r['PTS/G']}, PER: {r['PER']}, PII: {r['PII']:.3f}" for _, r in group.iterrows()],
                hoverinfo="text"
            ))

        buttons = [
            dict(label="All", method="update",
                 args=[{"visible": [True] * len(traces)}, {"title": "All Clusters"}])
        ]
        for i, label in enumerate(cluster_labels):
            vis = [j == i for j in range(len(traces))]
            buttons.append(dict(
                label=label,
                method="update",
                args=[{"visible": vis}, {"title": f"Cluster: {label}"}]
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.05, y=1.15)],
            title=dict(text="Player Clusters - PCA Projection", font=dict(size=24)),
            font=dict(size=14),
            plot_bgcolor="#fdfdfd",
            paper_bgcolor="#ffffff",
            margin=dict(l=60, r=40, t=90, b=60),
            xaxis=dict(title="PCA1", gridcolor="#e0e0e0", zeroline=False),
            yaxis=dict(title="PCA2", gridcolor="#e0e0e0", zeroline=False),
            hoverlabel=dict(bgcolor="white", bordercolor="black", font_size=13),
            hovermode="closest",
            legend_title=dict(text="Cluster")
        )
        fig.write_html(os.path.join(img_dir, "player_clusters_pca_interactive.html"))
        fig.write_image(os.path.join(img_dir, "player_clusters_pca.png"))
        fig.show()

    # --- EXPORT CLUSTERED DATASET ---
    output_dir = os.path.dirname(pkl_path)
    df_clean.to_pickle(os.path.join(output_dir, "All_time_Players_stats_clustered_labeled.pkl"))
    df_clean.to_excel(os.path.join(output_dir, "All_time_Players_stats_clustered_labeled.xlsx"), index=False)
    print(f"[INFO] Clustered dataset saved to: {output_dir}")

    return df_clean if return_data else None
