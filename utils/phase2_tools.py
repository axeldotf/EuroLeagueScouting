import os
import numpy as np
import pandas as pd
from scipy.stats import linregress

def logistic_threshold(age: float, k: float = 0.2, x0: float = 24, min_p: float = 0.3, max_p: float = 0.9) -> float:
    """
    Computes an age-adaptive threshold using a scaled logistic function.
    """
    return min_p + (max_p - min_p) / (1 + np.exp(-k * (age - x0)))

def dynamic_emergent_label(age: float, pii: float) -> bool:
    """
    Returns True if PII exceeds the dynamic threshold based on age.
    """
    return pii >= logistic_threshold(age)


def extract_career_emergents(
    input_path: str,
    output_path: str = None,
    min_improvement: int = 2,
    print_top_n: int = 10,
    show_table: bool = False,
    include_top_rookies: bool = True,
    max_age_first_season: int = 24
) -> pd.DataFrame:
    """
    Phase 2 - Career Trajectory-Based Pseudo-Labeling

    Identifies players who show strong career improvement trends across clusters and PII.
    Labels emergents based on trajectory types, slope, and optional rookie impact.
    """
    print(f"[INFO] Loading player-season data from: {input_path}")
    df = pd.read_pickle(input_path).copy()
    df["Season_Num"] = df["SEASON"].apply(lambda s: int(str(s).split("-")[0]))

    results = []

    for name, group in df.groupby("NAME"):
        group_sorted = group.sort_values("Season_Num")
        clusters = group_sorted["Cluster"].tolist()
        seasons = group_sorted["SEASON"].tolist()
        piis = group_sorted["PII"].tolist()
        season_nums = group_sorted["Season_Num"].tolist()

        if len(clusters) < 2:
            results.append((name, False, 0, None, None, None, "Insufficient"))
            continue

        best_cluster = min(clusters)
        worst_cluster = max(clusters)
        improvement = worst_cluster - best_cluster
        pii_gain = max(piis) - min(piis)
        slope, *_ = linregress(season_nums, piis)

        cluster_info = list(zip(seasons, clusters, piis))
        best_year = max([s for s, c, _ in cluster_info if c == best_cluster], default=None)
        worst_year = min([s for s, c, _ in cluster_info if c == worst_cluster], default=None)

        recovered = any(
            clusters[j] <= best_cluster
            for i in range(len(clusters)) if clusters[i] == worst_cluster
            for j in range(i + 1, len(clusters))
        )

        is_emergent = False
        traj_type = "NoChange"

        if improvement >= min_improvement and recovered:
            is_emergent = True
            if worst_cluster >= 3 and best_cluster <= 1:
                traj_type = "LowToHigh"
            elif worst_cluster == 2:
                traj_type = "MidToHigh"
            elif worst_cluster <= 1 and best_cluster == 0:
                traj_type = "HighToHigher"
            else:
                traj_type = "Other"
        elif pii_gain > 1.5 and slope > 0.3:
            is_emergent = True
            traj_type = "PIIonly"
        elif slope > 0.5:
            is_emergent = True
            traj_type = "UpwardTrend"
        elif clusters.count(0) >= 2:
            is_emergent = True
            traj_type = "ConsistentlyTop"

        results.append((name, is_emergent, improvement, worst_cluster, best_year, worst_year, traj_type))

    # === Build Summary Table ===
    df_summary = pd.DataFrame(results, columns=[
        "NAME", "Is_Career_Emergent", "Cluster_Improvement",
        "Worst_Cluster", "Best_Year", "Worst_Year", "Trajectory_Type"
    ])
    df_summary["Best_Cluster"] = df_summary["Worst_Cluster"] - df_summary["Cluster_Improvement"]

    # === Enrich with Stats ===
    pii_stats = df.groupby("NAME")["PII"].agg(
        Mean_PII="mean", Max_PII="max", Min_PII="min", Std_PII="std"
    ).round(4)
    season_counts = df.groupby("NAME")["SEASON"].nunique().rename("Seasons_Played")
    first_seasons = df.groupby("NAME")["SEASON"].min().rename("First_Season")

    df_summary = (
        df_summary
        .merge(first_seasons, on="NAME", how="left")
        .merge(pii_stats, on="NAME", how="left")
        .merge(season_counts, on="NAME", how="left")
    )

    # === Optional: Add Rookie Emergence ===
    if include_top_rookies and "AGE" in df.columns:
        df["AGE"] = df["AGE"].astype(float)
        first_season_map = df.groupby("NAME")["Season_Num"].min()
        rookies = df[df["Season_Num"] == df["NAME"].map(first_season_map)].copy()

        standout_rookies = rookies[
            rookies.apply(lambda r: dynamic_emergent_label(r["AGE"], r["PII"]), axis=1)
        ]
        standout_names = standout_rookies["NAME"]

        df_summary.loc[df_summary["NAME"].isin(standout_names), "Is_Career_Emergent"] = True
        df_summary.loc[df_summary["NAME"].isin(standout_names), "Trajectory_Type"] = "RookieImpact"

    # === Compute Emergence Score ===
    df_summary["Emergence_Score"] = (
        df_summary["Cluster_Improvement"] *
        (df_summary["Mean_PII"] + df_summary["Max_PII"]) /
        (1 + df_summary["Std_PII"])
    ).round(3)
    df_summary.loc[~df_summary["Is_Career_Emergent"], "Emergence_Score"] = 0.0

    # === Save to Output ===
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_summary.to_pickle(output_path)
        print(f"[INFO] Emergence summary saved to: {output_path}")

    # === Display Top Players ===
    if show_table:
        top_emergents = (
            df_summary[df_summary["Is_Career_Emergent"]]
            .sort_values(by="Emergence_Score", ascending=False)
            .loc[:, ["NAME", "Best_Year", "Worst_Year", "Emergence_Score", "Trajectory_Type"]]
            .head(print_top_n)
        )
        print(f"\n[PHASE 2] Top {print_top_n} career emergents:\n")
        print(top_emergents.to_string(index=False))

    # === Log Summary ===
    total = len(df_summary)
    emergents = df_summary["Is_Career_Emergent"].sum()
    print(f"[INFO] Players analyzed: {total}")
    print(f"[INFO] Career-level emergents detected: {emergents} ({(emergents / total * 100):.2f}%)")

    return df_summary
