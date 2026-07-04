"""Model training pipeline — 5 models, hyperparameter tuning, calibration."""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

from src.config import (
    ALL_MODELS_PKL, BEST_MODEL_PKL, FEATURE_COLS_PKL,
    METADATA_JSON, PERM_IMP_CSV, CLASS_NAMES, CLASS_TO_IDX,
)
from src.feature_engineering import build_training_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = ["home_confederation", "away_confederation"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions with search spaces
# ─────────────────────────────────────────────────────────────────────────────

def get_model_specs():
    specs = {}

    # 1. Logistic Regression
    specs["LogisticRegression"] = {
        "model": LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                    max_iter=1000, random_state=42),
        "params": {
            "clf__C":           [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "clf__class_weight": [None, "balanced"],
        },
    }

    # 2. Decision Tree
    specs["DecisionTree"] = {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "clf__max_depth":        [4, 6, 8, 10, 12, None],
            "clf__min_samples_leaf": [10, 20, 30, 50],
            "clf__min_samples_split":[20, 40, 60],
            "clf__class_weight":     [None, "balanced"],
        },
    }

    # 3. KNN
    specs["KNN"] = {
        "model": KNeighborsClassifier(),
        "params": {
            "clf__n_neighbors": [15, 21, 31, 41, 51, 71],
            "clf__weights":     ["uniform", "distance"],
            "clf__metric":      ["euclidean", "manhattan"],
        },
    }

    # 4. XGBoost
    try:
        from xgboost import XGBClassifier
        specs["XGBoost"] = {
            "model": XGBClassifier(
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", random_state=42,
                tree_method="hist", device="cpu", verbosity=0,
            ),
            "params": {
                "clf__n_estimators":  [200, 300, 400, 500],
                "clf__max_depth":     [3, 4, 5, 6],
                "clf__learning_rate": [0.02, 0.05, 0.08, 0.1],
                "clf__subsample":     [0.7, 0.8, 0.9],
                "clf__colsample_bytree": [0.7, 0.8, 0.9],
                "clf__reg_alpha":     [0.0, 0.1, 0.5],
                "clf__reg_lambda":    [1.0, 2.0, 3.0],
                "clf__min_child_weight": [3, 5, 10],
            },
        }
    except ImportError:
        pass

    # 5. CatBoost
    try:
        from catboost import CatBoostClassifier
        specs["CatBoost"] = {
            "model": CatBoostClassifier(
                loss_function="MultiClass", eval_metric="Accuracy",
                random_seed=42, verbose=0, thread_count=-1,
            ),
            "params": {
                "clf__iterations":     [300, 500, 700],
                "clf__depth":          [4, 5, 6, 7],
                "clf__learning_rate":  [0.03, 0.05, 0.08, 0.1],
                "clf__l2_leaf_reg":    [1, 3, 5, 7],
                "clf__bagging_temperature": [0.5, 1.0, 1.5],
            },
        }
    except ImportError:
        pass

    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Train single model
# ─────────────────────────────────────────────────────────────────────────────

def train_single(
    name: str,
    spec: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weights: np.ndarray,
    feature_cols: list[str],
    n_iter: int = 40,
) -> tuple[Pipeline, dict]:
    print(f"  ▶ Training {name}...")

    preprocessor = build_preprocessor(X_train[feature_cols])
    pipe = Pipeline([("pre", preprocessor), ("clf", spec["model"])])

    # Remap param names to include pipeline step prefix
    params = {}
    for k, v in spec["params"].items():
        params[k] = v

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        search = RandomizedSearchCV(
            pipe, params, n_iter=n_iter, cv=cv,
            scoring="neg_log_loss", n_jobs=-1,
            random_state=42, verbose=0, refit=True,
        )
        # CatBoost / XGBoost may support sample_weight via fit_params
        fit_params = {}
        if name not in ("KNN",):
            fit_params["clf__sample_weight"] = sample_weights

        try:
            search.fit(X_train[feature_cols], y_train, **fit_params)
        except TypeError:
            search.fit(X_train[feature_cols], y_train)

        best_pipe = search.best_estimator_
    except Exception as e:
        print(f"    RandomizedSearch failed ({e}), using default params")
        try:
            if name not in ("KNN",):
                pipe.fit(X_train[feature_cols], y_train,
                         clf__sample_weight=sample_weights)
            else:
                pipe.fit(X_train[feature_cols], y_train)
        except TypeError:
            pipe.fit(X_train[feature_cols], y_train)
        best_pipe = pipe

    # Evaluate
    y_pred    = best_pipe.predict(X_val[feature_cols])
    y_proba   = best_pipe.predict_proba(X_val[feature_cols])
    acc       = accuracy_score(y_val, y_pred)
    ll        = log_loss(y_val, y_proba, labels=range(len(CLASS_NAMES)))

    print(f"    ✓ {name}: acc={acc:.4f}  log_loss={ll:.4f}")

    metrics = {
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "report": classification_report(y_val, y_pred, output_dict=True),
    }
    return best_pipe, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_all(test_fraction: float = 0.15, n_iter: int = 40):
    print("=" * 60)
    print("FIFA WC 2026 — Model Training Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Building feature matrix...")
    X, y, weights = build_training_dataset()
    y = y.map(CLASS_TO_IDX)
    print(f"  Dataset: {len(X):,} samples | {X.shape[1]} raw features")
    print(f"  Class balance: {y.value_counts().to_dict()}")

    # Feature columns
    feature_cols = [c for c in X.columns]

    # 2. Chronological train/val split
    print("\n[2/6] Splitting data (chronological)...")
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_val   = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val   = y.iloc[:split_idx], y.iloc[split_idx:]
    w_train           = weights[:split_idx]
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    # 3. Train models
    print("\n[3/6] Training models...")
    specs       = get_model_specs()
    all_models  = {}
    all_metrics = {}

    for name, spec in specs.items():
        model, metrics = train_single(
            name, spec, X_train, y_train, X_val, y_val,
            w_train, feature_cols, n_iter=n_iter,
        )
        all_models[name]  = model
        all_metrics[name] = metrics

    # 4. Pick best model (by accuracy on val set)
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["accuracy"])
    best_model = all_models[best_name]
    print(f"\n[4/6] Best model: {best_name} "
          f"(acc={all_metrics[best_name]['accuracy']:.4f})")

    # 5. Calibrate probabilities
    print("\n[5/6] Calibrating probabilities...")
    try:
        # Need raw estimator from pipeline to calibrate
        # We calibrate the full pipeline's outputs using CalibratedClassifierCV
        # with cv="prefit" on the validation set
        from sklearn.calibration import CalibratedClassifierCV
        calibrated = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
        calibrated.fit(X_val[feature_cols], y_val)
        final_model = calibrated

        # Re-evaluate calibrated model
        y_proba_cal = calibrated.predict_proba(X_val[feature_cols])
        ll_cal = log_loss(y_val, y_proba_cal, labels=calibrated.classes_)
        print(f"  Calibrated log_loss: {ll_cal:.4f}")
    except Exception as e:
        print(f"  Calibration failed ({e}), using uncalibrated model")
        final_model = best_model

    # 6. Permutation importance on best model
    print("\n[6/6] Computing feature importance...")
    try:
        preprocessor = best_model.named_steps["pre"]
        X_val_trans   = preprocessor.transform(X_val[feature_cols])
        feature_names = preprocessor.get_feature_names_out()

        clf = best_model.named_steps["clf"]
        perm = permutation_importance(
            clf, X_val_trans, y_val,
            n_repeats=10, random_state=42,
            scoring="accuracy", n_jobs=-1,
        )
        perm_df = pd.DataFrame({
            "feature":          feature_names,
            "importance_mean":  perm.importances_mean,
            "importance_std":   perm.importances_std,
        }).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(PERM_IMP_CSV, index=False)
        print(f"  Saved permutation importance → {PERM_IMP_CSV}")
    except Exception as e:
        print(f"  Importance failed: {e}")
        perm_df = pd.DataFrame()

    # Save artifacts
    print("\nSaving artifacts...")
    with open(BEST_MODEL_PKL,   "wb") as f: pickle.dump(final_model, f)
    with open(ALL_MODELS_PKL,   "wb") as f: pickle.dump(all_models,  f)
    with open(FEATURE_COLS_PKL, "wb") as f: pickle.dump(feature_cols, f)

    metadata = {
        "best_model_name": best_name,
        "all_models": {k: {
            "accuracy": v["accuracy"],
            "log_loss": v["log_loss"],
        } for k, v in all_metrics.items()},
        "validation_metrics": all_metrics[best_name],
        "feature_count": len(feature_cols),
        "train_samples": len(X_train),
        "val_samples":   len(X_val),
        "class_names":   CLASS_NAMES,
    }
    with open(METADATA_JSON, "w") as f: json.dump(metadata, f, indent=2)

    print(f"\n✅ Training complete! All artifacts saved to {BEST_MODEL_PKL.parent}")
    print("\nModel comparison:")
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1]["accuracy"]):
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:22s}  acc={m['accuracy']:.4f}  log_loss={m['log_loss']:.4f}{marker}")

    return final_model, feature_cols, all_metrics
