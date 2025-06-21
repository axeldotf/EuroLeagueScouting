import pandas as pd
import os
import re
from utils.phase1_tools import compute_pii
import joblib

def preprocess_season_dataset(
    path_input: str,
    path_reference: str,
    output_dir: str = "output",
    verbose: bool = True,
    age_for_is_under: int = 25
) -> pd.DataFrame:
    """
    Phase 5 - Preprocess raw season data for model inference.

    - Infers season from filename
    - Cleans player data and filters out unreliable entries
    - Computes percentile ranks and PII score
    - Adds Is_U20 flag

    Returns:
    - Preprocessed DataFrame
    """
    def load_data(path: str) -> pd.DataFrame:
        if path.endswith(".pkl") or os.path.exists(path.replace(".xlsx", ".pkl")):
            return pd.read_pickle(path.replace(".xlsx", ".pkl"))
        return pd.read_excel(path)

    df = load_data(path_input)
    df_ref = load_data(path_reference)

    match = re.search(r"(\d{2})[_\-](\d{2})", os.path.basename(path_input))
    if match:
        y1, y2 = int(match.group(1)), int(match.group(2))
        full_y1 = 2000 + y1 if y1 < 50 else 1900 + y1
        full_y2 = 2000 + y2 if y2 < 50 else 1900 + y2
        season_str = f"{full_y1}-{full_y2}"
    else:
        season_str = "Unknown"

    if "SEASON" in df.columns:
        df.drop(columns=["SEASON"], inplace=True)
    df.insert(2, "SEASON", season_str)

    if verbose:
        print(f"[INFO] Season inferred from filename: {season_str}")

    valid_columns = [col for col in df.columns if col in df_ref.columns]
    df_cleaned = df[valid_columns].copy()

    if "GP" in df_cleaned.columns:
        before = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned["GP"] >= 5]
        if verbose:
            print(f"[INFO] Removed {before - len(df_cleaned)} players with GP < 5")

    df_cleaned["AGE"] = pd.to_numeric(df_cleaned["AGE"], errors="coerce")
    df_cleaned = df_cleaned[df_cleaned["AGE"] <= 45]

    features_for_pii = [
        'PTS/G', 'USG%', 'Min/G', 'TS%', 'eFG%', 'PER',
        'NET RTG', 'W%', 'AST/TO', 'AST%', 'TO Ratio',
        'DEF RTG', 'ST%', 'BLK%', 'TR%', "AGE"
    ]
    missing = [col for col in features_for_pii if col not in df_cleaned.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing required features for PII: {missing}")

    for col in features_for_pii:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
        df_cleaned[f"rnk_{col}"] = df_cleaned[col].rank(pct=True)

    df_cleaned["PII"] = compute_pii(df_cleaned, age_for_is_under)
    df_cleaned["Is_U20"] = (df_cleaned["AGE"] < 20).astype(int)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(path_input))[0]
    df_cleaned.to_excel(os.path.join(output_dir, f"{base_name}_preprocessed.xlsx"), index=False)
    df_cleaned.to_pickle(os.path.join(output_dir, f"{base_name}_preprocessed.pkl"))

    if verbose:
        print(f"[INFO] Saved preprocessed data to: {output_dir}")
    return df_cleaned

def apply_trained_model_to_season(
    input_path: str,
    output_path: str = None,
    model_name: str = "rf",
    models_dir: str = "models/rf",
    threshold: float = 0.2,
    optimize_threshold: bool = False,
    max_age: int = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Phase 5 - Apply trained model to current season data.

    - Loads model and feature list
    - Applies dynamic age-based thresholding (logistic)
    - Returns sorted prediction DataFrame
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import joblib
    import pandas as pd
    import os

    def load_input(path: str) -> pd.DataFrame:
        return pd.read_pickle(path) if path.endswith(".pkl") else pd.read_excel(path)

    def logistic_threshold(age: float, min_p: float, max_p: float, k: float = 0.2, x0: float = 24):
        return min_p + (max_p - min_p) / (1 + np.exp(-k * (age - x0)))

    model_path = os.path.join(models_dir, f"model_{model_name}.joblib")
    model = joblib.load(model_path)
    if verbose:
        print(f"[INFO] Model loaded from: {model_path}")

    df = load_input(input_path)
    if verbose:
        print(f"[INFO] Data loaded from: {input_path}")

    if max_age is not None:
        df = df[df["AGE"] <= max_age].copy()

    if "AGE" in df.columns and "Is_U20" not in df.columns:
        df.loc[:, "Is_U20"] = (df["AGE"] < 20).astype(int)

    features_path = os.path.join(models_dir, "model_features.txt")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"[ERROR] Missing feature list: {features_path}")
    with open(features_path) as f:
        feature_names = [line.strip() for line in f.readlines()]

    meta_cols = ["NAME", "SEASON", "PII", "Cluster", "Cluster_Label", "AGE", "TEAM"]
    if all(c in df.columns for c in meta_cols):
        X_meta = df[meta_cols].copy()
    else:
        X_meta = df[["NAME", "AGE", "SEASON", "TEAM"]].copy()

    X = df[feature_names].copy().apply(pd.to_numeric, errors="coerce")

    if model_name == "logreg":
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    y_prob = model.predict_proba(X)[:, 1]
    min_p, max_p = float(np.min(y_prob)), float(np.max(y_prob))

    def dynamic_label(age, prob):
        if age >= 32:
            return prob >= 0.9  # stricter fixed threshold for veterans
        return prob >= logistic_threshold(age, min_p, max_p)

    y_pred = [dynamic_label(age, prob) for age, prob in zip(df["AGE"], y_prob)]

    df_results = X_meta.copy()
    df_results.loc[:, "prob_emergent"] = np.round(y_prob, 3)
    df_results.loc[:, "predicted_emergent"] = y_pred
    df_results = df_results.sort_values("prob_emergent", ascending=False)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        base = os.path.splitext(output_path)[0]
        df_results.to_excel(f"{base}.xlsx", index=False)
        df_results.to_pickle(f"{base}.pkl")
        if verbose:
            print(f"[INFO] Predictions saved to:\n       - {base}.xlsx\n       - {base}.pkl")

    return df_results

def plot_emergent_predictions_interactive(
    df_results: pd.DataFrame,
    output_dir: str = "img",
    base_filename: str = "emergent_predictions"
):
    """
    Phase 5 - Interactive plot of model predictions by player age and probability.

    Allows toggling between emerging / non-emerging players.
    """
    import plotly.graph_objects as go
    os.makedirs(output_dir, exist_ok=True)

    df = df_results.copy()
    df["Color"] = df["predicted_emergent"].map({True: "#1f77b4", False: "#d3d3d3"}).fillna("#d3d3d3")

    traces = []
    for label, is_emergent in [("Emerging", True), ("Not Emerging", False), ("All", None)]:
        sub_df = df if is_emergent is None else df[df["predicted_emergent"] == is_emergent]
        traces.append(go.Scatter(
            x=sub_df["AGE"],
            y=sub_df["prob_emergent"],
            mode="markers",
            name=label,
            marker=dict(size=12, color=sub_df["Color"], opacity=0.75, line=dict(width=1.5, color="black")),
            hovertext=[
                f"<b>{r['NAME']}</b><br>Team: {r['TEAM']}<br>Age: {r['AGE']}<br>Prob: {r['prob_emergent']:.2f}"
                for _, r in sub_df.iterrows()
            ],
            hoverinfo="text",
            visible=(label == "All")
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Emergent Probability - Season Application",
        font=dict(size=14),
        plot_bgcolor="#fdfdfd",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=40, t=90, b=60),
        xaxis=dict(title="Age", gridcolor="#e0e0e0", zeroline=False),
        yaxis=dict(title="Probability", gridcolor="#e0e0e0", zeroline=False, range=[0, 1.0]),
        hoverlabel=dict(bgcolor="white", bordercolor="black", font_size=13),
        hovermode="closest",
        updatemenus=[dict(
            buttons=[
                dict(label="All", method="update", args=[{"visible": [False, False, True]}]),
                dict(label="Emerging", method="update", args=[{"visible": [True, False, False]}]),
                dict(label="Not Emerging", method="update", args=[{"visible": [False, True, False]}]),
            ],
            direction="down", showactive=True, x=0.85, y=1.1
        )]
    )

    html_path = os.path.join(output_dir, f"{base_filename}_interactive.html")
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    fig.write_html(html_path)
    fig.write_image(png_path)

    print(f"[INFO] Interactive plot saved to:\n       - {html_path}\n       - {png_path}")
    fig.show()
