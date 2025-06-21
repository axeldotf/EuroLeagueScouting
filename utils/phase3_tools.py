from typing import Tuple
import os
import pandas as pd
import plotly.graph_objects as go

def build_supervised_dataset(
    pkl_path: str,
    df_emergents: pd.DataFrame,
    max_age: int = None,
    young_age_threshold: int = 23,
    label_mode: str = "seasonal"  # "seasonal" or "career"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Phase 3 - Build Supervised Dataset for Predictive Modeling

    Constructs a dataset using season-level stats and pseudo-labels (from Phase 2).
    Suitable for training models to predict player emergence in t+1 or over career.

    Parameters:
    - pkl_path: Path to .pkl file with player-season data (with PII and clusters).
    - df_emergents: DataFrame with pseudo-labels from Phase 2 (career emergence).
    - max_age: If set, filters players older than this age.
    - young_age_threshold: Used for binary age flag (e.g. U23).
    - label_mode: 
        - "seasonal": Label as 1 if emergence occurs next season (t+1).
        - "career": Label as 1 if emergence occurs anytime before best season.

    Returns:
    - X_features: DataFrame with model-ready features.
    - y_labels: Binary label series.
    """
    df = pd.read_pickle(pkl_path).copy()
    df = df[(df["PER"] <= 40) & (df["NET RTG"] <= 100)]
    if max_age is not None:
        df = df[df["AGE"] <= max_age]

    df["Season_Start_Year"] = df["SEASON"].apply(lambda x: int(str(x).split("-")[0]))
    df["Key"] = df["NAME"] + "_" + df["Season_Start_Year"].astype(str)

    if label_mode == "seasonal":
        df["Next_Season"] = df["Season_Start_Year"] + 1
        df["Next_Key"] = df["NAME"] + "_" + df["Next_Season"].astype(str)
        df = df[df["Next_Key"].isin(set(df["Key"]))]

    df["Is_Under"] = (df["AGE"] <= young_age_threshold).astype(int)

    base_features = [
        'PTS/G', 'USG%', 'Min/G', 'TS%', 'eFG%', 'PER',
        'NET RTG', 'W%', 'AST/TO', 'AST%', 'TO Ratio',
        'DEF RTG', 'ST%', 'BLK%', 'TR%', 'AGE'
    ]
    debug_columns = ["NAME", "SEASON", "PII", "TEAM", "Cluster", "Cluster_Label"]

    # Add deltas of selected key metrics
    df.sort_values(by=["NAME", "Season_Start_Year"], inplace=True)
    for col in ['PTS/G', 'PER', 'USG%']:
        df[f"Δ_{col}"] = df.groupby("NAME")[col].diff()
    df.dropna(subset=[f"Δ_{col}" for col in ['PTS/G', 'PER', 'USG%']], inplace=True)

    # Filter emergents with known breakout year
    df_emergents = df_emergents[df_emergents["Is_Career_Emergent"] & df_emergents["Best_Year"].notna()].copy()
    df_emergents["Best_Year_Num"] = df_emergents["Best_Year"].apply(lambda x: int(str(x).split("-")[0]))

    if label_mode == "seasonal":
        df_emergents["Emergence_Season"] = df_emergents["Best_Year_Num"] - 1
        emergent_keys = set(df_emergents["NAME"] + "_" + df_emergents["Emergence_Season"].astype(str))
        df["Label"] = df["Key"].apply(lambda k: k in emergent_keys)
    elif label_mode == "career":
        df = df.merge(df_emergents[["NAME", "Best_Year_Num"]], on="NAME", how="left")
        df["Label"] = (df["Season_Start_Year"] <= df["Best_Year_Num"]).fillna(False)
    else:
        raise ValueError(f"[ERROR] Unsupported label_mode: {label_mode}")

    X_features = df[debug_columns + base_features].copy()
    y_labels = df["Label"]

    return X_features, y_labels


def plot_top_players_by_metric(
    X_features: pd.DataFrame,
    y_labels: pd.Series,
    metrics: list,
    cutoff: int = 100,
    output_dir: str = "img"
):
    """
    Plots top players by various metrics vs. age, highlighting emergents.

    Parameters:
    - X_features: DataFrame with player metrics.
    - y_labels: Series of binary pseudo-labels (1 = emerging).
    - metrics: List of metrics to visualize.
    - cutoff: Number of top players to show per metric.
    - output_dir: Directory to save HTML and PNG visualizations.
    """
    if "AGE" not in X_features.columns or "NAME" not in X_features.columns:
        raise ValueError("X_features must include 'AGE' and 'NAME'.")

    df_plot = X_features.copy()
    df_plot["Emerging"] = y_labels.astype(int)
    df_plot["Color"] = df_plot["Emerging"].map({1: "#1f77b4", 0: "#d3d3d3"})

    traces, buttons = [], []

    for i, metric in enumerate(metrics):
        if metric not in df_plot.columns:
            print(f"[WARNING] Metric '{metric}' not found. Skipping.")
            continue

        top_df = df_plot.sort_values(by=metric, ascending=False).head(cutoff)

        trace = go.Scatter(
            x=top_df["AGE"],
            y=top_df[metric],
            mode="markers",
            name=metric,
            marker=dict(
                size=12,
                color=top_df["Color"],
                opacity=0.75,
                line=dict(width=1.5, color="black")
            ),
            hovertext=[
                f"<b>{row['NAME']}</b><br>Age: {row['AGE']}<br>{metric}: {row[metric]:.2f}<br>Emerging: {'Yes' if row['Emerging'] else 'No'}"
                for _, row in top_df.iterrows()
            ],
            hoverinfo="text",
            visible=(i == 0)
        )
        traces.append(trace)
        buttons.append(dict(
            label=metric,
            method="update",
            args=[
                {"visible": [j == i for j in range(len(metrics))]},
                {"title": f"Top {cutoff} Players - {metric} vs Age"}
            ]
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.05, y=1.15)],
        title=dict(text=f"Top {cutoff} Players - {metrics[0]} vs Age", font=dict(size=24)),
        font=dict(size=14),
        plot_bgcolor="#fdfdfd",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=40, t=90, b=60),
        xaxis=dict(title="Age", gridcolor="#e0e0e0"),
        yaxis=dict(title="Metric Value", gridcolor="#e0e0e0"),
        hoverlabel=dict(bgcolor="white", bordercolor="black", font_size=13),
        hovermode="closest"
    )

    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, f"top{cutoff}_metrics_scatter_interactive.html"))
    fig.write_image(os.path.join(output_dir, f"top{cutoff}_metrics_scatter.png"))

    print(f"[INFO] Saved interactive plot to: {output_dir}")
    fig.show()
