from typing import Tuple
import os
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def train_and_evaluate_model_simple(
    X_features: pd.DataFrame,
    y_labels: pd.Series,
    model_name: str = "rf",
    model=None,
    test_size: float = 0.3,
    threshold: float = 0.2,
    use_smote: bool = False,
    random_state: int = 42,
    verbose: bool = True,
    save_model_path: str = None,
    optimize_threshold: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 4 - Supervised Training (Simple Mode)

    Trains and evaluates a classification model to predict player emergence.
    Supports threshold optimization and SMOTE balancing.

    Returns:
    - Results DataFrame (with probabilities and predictions)
    - Feature importance DataFrame
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    meta_cols = ["NAME", "SEASON", "PII", "Cluster", "Cluster_Label", "AGE", "TEAM"]
    X = X_features.drop(columns=meta_cols, errors="ignore")
    meta = X_features[meta_cols] if all(c in X_features.columns for c in meta_cols) else None
    X = X.apply(pd.to_numeric, errors="coerce")

    if model_name == "logreg":
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    # split = train_test_split(X, y_labels, meta, test_size=test_size, stratify=y_labels, random_state=random_state) \
    #     if meta is not None else train_test_split(X, y_labels, test_size=test_size, stratify=y_labels, random_state=random_state)
    # X_train, X_test, y_train, y_test, *meta_split = split
    # meta_train, meta_test = meta_split if meta else (None, None)

    split = train_test_split(X, y_labels, meta, test_size=test_size, stratify=y_labels, random_state=random_state) \
        if meta is not None else train_test_split(X, y_labels, test_size=test_size, stratify=y_labels, random_state=random_state)
    X_train, X_test, y_train, y_test, *meta_split = split
    meta_train, meta_test = meta_split if meta is not None else (None, None)

    if use_smote:
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    if model is None:
        if model_name == "rf":
            model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state)
        elif model_name == "logreg":
            model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=random_state)
        elif model_name == "xgb":
            model = XGBClassifier(eval_metric="logloss", random_state=random_state)
        else:
            raise ValueError(f"[ERROR] Unsupported model: '{model_name}'")

    model.fit(X_train, y_train)

    if save_model_path:
        os.makedirs(save_model_path, exist_ok=True)
        joblib.dump(model, os.path.join(save_model_path, f"model_{model_name}.joblib"))
        with open(os.path.join(save_model_path, "model_features.txt"), "w") as f:
            f.writelines([col + "\n" for col in X.columns])
        if verbose:
            print(f"[INFO] Model and features saved in '{save_model_path}'")

    y_prob = model.predict_proba(X_test)[:, 1]

    if optimize_threshold:
        prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = f1_scores.argmax()
        threshold = thresholds[best_idx]
        if verbose:
            print(f"[INFO] Optimal threshold for F1: {threshold:.3f} (F1 = {f1_scores[best_idx]:.4f})")

    y_pred = (y_prob >= threshold)

    if verbose:
        print(f"[INFO] Model: {model_name.upper()} | Threshold: {threshold:.2f}")
        print(classification_report(y_test, y_pred))
        print(f"[INFO] AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    # === Feature Importance ===
    feature_importance_df = None
    try:
        if model_name in ["rf", "xgb"]:
            feature_importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
        elif model_name == "logreg":
            coeffs = model.coef_[0]
            feature_importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": np.abs(coeffs),
                "Coefficient": coeffs
            }).sort_values(by="Importance", ascending=False)

        if verbose and feature_importance_df is not None:
            print("[INFO] Top 10 Feature Importances:")
            print(feature_importance_df.head(10).to_string(index=False))

            top_k = min(10, len(feature_importance_df))
            plt.figure(figsize=(5, 5))
            plt.barh(
                feature_importance_df["Feature"][:top_k][::-1],
                feature_importance_df["Importance"][:top_k][::-1],
                color="steelblue", edgecolor="black"
            )
            plt.xlabel("Importance Score")
            plt.title(f"Top {top_k} Features - {model_name.upper()}")
            plt.tight_layout()
            plt.show()

        if save_model_path and feature_importance_df is not None:
            out_path = os.path.join(save_model_path, f"feature_importance_{model_name}.xlsx")
            feature_importance_df.to_excel(out_path, index=False)
            if verbose:
                print(f"[INFO] Feature importance saved to: {out_path}")

    except Exception as e:
        print(f"[WARNING] Feature importance failed: {e}")

    results = pd.DataFrame({
        "prob_emergent": y_prob,
        "y_true": y_test.values,
        "y_pred": y_pred
    })
    if meta_test is not None:
        results = pd.concat([meta_test.reset_index(drop=True), results], axis=1)

    return results.sort_values("prob_emergent", ascending=False), feature_importance_df

def train_and_evaluate_model_optimized(
    X_features: pd.DataFrame,
    y_labels: pd.Series,
    model_name: str = "rf",
    use_smote: bool = False,
    test_size: float = 0.3,
    random_state: int = 42,
    optimize_threshold: bool = True,
    save_model_path: str = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 4 - Supervised Training (Optimized via GridSearch)

    Performs hyperparameter tuning, optional SMOTE balancing, and full evaluation
    of model performance. Supports Random Forest, Logistic Regression, XGBoost.

    Returns:
    - Sorted DataFrame with results
    - Feature importance DataFrame
    """
    from sklearn.model_selection import GridSearchCV

    warnings.filterwarnings("ignore", category=UserWarning)

    meta_cols = ["NAME", "SEASON", "PII", "Cluster", "Cluster_Label", "AGE", "TEAM"]
    X = X_features.drop(columns=meta_cols, errors="ignore")
    meta = X_features[meta_cols] if all(c in X_features.columns for c in meta_cols) else None
    X = X.apply(pd.to_numeric, errors="coerce")

    if model_name == "logreg":
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y_labels, meta, test_size=test_size, stratify=y_labels, random_state=random_state
    )

    if use_smote:
        X_train, y_train = SMOTE(k_neighbors=3, random_state=random_state).fit_resample(X_train, y_train)
        if verbose:
            print(f"[INFO] SMOTE applied: {len(X_train)} samples after resampling")

    # Grid search setup
    if model_name == "rf":
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"]
        }
        base_model = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    elif model_name == "logreg":
        param_grid = {
            "C": [0.1, 1, 10],
            "penalty": ["l2"]
        }
        base_model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=random_state)
    elif model_name == "xgb":
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.3]
        }
        base_model = XGBClassifier(eval_metric="logloss", random_state=random_state)
    else:
        raise ValueError(f"[ERROR] Unsupported model: {model_name}")

    grid = GridSearchCV(base_model, param_grid, scoring="f1", cv=5, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    if verbose:
        print(f"[INFO] Best {model_name.upper()} parameters: {grid.best_params_}")

    if save_model_path:
        os.makedirs(save_model_path, exist_ok=True)
        joblib.dump(model, os.path.join(save_model_path, f"model_{model_name}.joblib"))
        with open(os.path.join(save_model_path, "model_features.txt"), "w") as f:
            f.writelines([col + "\n" for col in X.columns])
        if verbose:
            print(f"[INFO] Model and feature list saved to '{save_model_path}'")

    y_prob = model.predict_proba(X_test)[:, 1]

    threshold = 0.5
    if optimize_threshold:
        prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = f1_scores.argmax()
        threshold = thresholds[best_idx]
        if verbose:
            print(f"[INFO] Optimal F1-threshold: {threshold:.3f} (F1 = {f1_scores[best_idx]:.4f})")

    y_pred = (y_prob >= threshold)

    if verbose:
        print(f"[RESULT] Final Evaluation - {model_name.upper()}")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    # Feature importance
    feature_importance_df = None
    try:
        if model_name in ["rf", "xgb"]:
            feature_importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
        elif model_name == "logreg":
            coeffs = model.coef_[0]
            feature_importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": np.abs(coeffs),
                "Coefficient": coeffs
            }).sort_values(by="Importance", ascending=False)

        if verbose and feature_importance_df is not None:
            print("[INFO] Top 10 Feature Importances:")
            print(feature_importance_df.head(10).to_string(index=False))

            top_k = min(10, len(feature_importance_df))
            plt.figure(figsize=(5, 5))
            plt.barh(
                feature_importance_df["Feature"][:top_k][::-1],
                feature_importance_df["Importance"][:top_k][::-1],
                color="steelblue", edgecolor="black"
            )
            plt.xlabel("Importance Score")
            plt.title(f"Top {top_k} Features - {model_name.upper()}")
            plt.tight_layout()
            plt.show()

        if save_model_path and feature_importance_df is not None:
            out_path = os.path.join(save_model_path, f"feature_importance_{model_name}.xlsx")
            feature_importance_df.to_excel(out_path, index=False)
            if verbose:
                print(f"[INFO] Feature importance saved to: {out_path}")
    except Exception as e:
        print(f"[WARNING] Feature importance failed: {e}")

    results = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "prob_emergent": y_prob
    })
    if meta_test is not None:
        results = pd.concat([meta_test.reset_index(drop=True), results], axis=1)

    return results.sort_values("prob_emergent", ascending=False), feature_importance_df

def plot_model_comparison_dropdown(
    results_dict: dict,
    output_dir: str = "img/model_comparisons",
    base_filename: str = "emergents_model_comparison"
):
    """
    Phase 4 - Visualize Emergence Predictions Across Models (Dropdown)

    Generates an interactive dropdown scatter plot comparing models’ predicted
    probabilities of emergence per player.
    """
    import plotly.graph_objects as go

    traces = []
    buttons = []
    model_names = list(results_dict.keys())

    for i, model in enumerate(model_names):
        df = results_dict[model].copy()
        df["Emergent_Color"] = df["y_true"].map({True: '#1f77b4', False: '#d3d3d3'})

        trace = go.Scatter(
            x=df["AGE"],
            y=df["prob_emergent"],
            mode="markers",
            name=model.upper(),
            marker=dict(size=12, color=df["Emergent_Color"], opacity=0.75, line=dict(width=1.5, color='black')),
            text=[
                f"<b>{row['NAME']}</b> ({row['SEASON']})<br>"
                f"Probability: <b>{row['prob_emergent']:.2f}</b><br>"
                f"Cluster: {row['Cluster_Label']}"
                for _, row in df.iterrows()
            ],
            hoverinfo="text",
            visible=(i == 0)
        )
        traces.append(trace)

        buttons.append(dict(
            label=model.upper(),
            method="update",
            args=[{"visible": [j == i for j in range(len(model_names))]},
                  {"title": f"Predicted Emergent Players - {model.upper()}"}]
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.05, y=1.15)],
        title=dict(text=f"Predicted Emergent Players - {model_names[0].upper()}", font=dict(size=24)),
        font=dict(size=14),
        plot_bgcolor="#fdfdfd",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=40, t=90, b=60),
        xaxis=dict(title="Age", gridcolor="#e0e0e0", zeroline=False),
        yaxis=dict(title="Emergence Probability", gridcolor="#e0e0e0", range=[0, 1.0], zeroline=False),
        hoverlabel=dict(bgcolor="white", bordercolor="black", font_size=13),
        hovermode="closest"
    )

    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, f"{base_filename}_interactive.html"))
    fig.write_image(os.path.join(output_dir, f"{base_filename}.png"))
    print(f"[INFO] Saved dropdown plot to {output_dir}")
    fig.show()

def plot_model_comparison_bars(model_scores: dict, title: str = "Model Performance Comparison"):
    """
    Phase 4 - Bar Plot of Model Metrics

    Creates a 2x2 grid bar chart comparing F1, Precision, Recall, and AUC-ROC for multiple models.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(model_scores).T.reset_index().rename(columns={"index": "Model"})

    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    fig.suptitle(title, fontsize=16)

    metrics = ["F1_True", "Precision_True", "Recall_True", "AUC_ROC"]
    titles = ["F1 Score (True Class)", "Precision (True Class)", "Recall (True Class)", "AUC-ROC"]

    for ax, metric, t in zip(axs.flat, metrics, titles):
        ax.bar(df["Model"], df[metric], color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black")
        ax.set_title(t)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
