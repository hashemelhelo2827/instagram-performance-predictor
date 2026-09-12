"""
Social Media Performance Predictor
-----------------------------------
Full pipeline covering the three modeling tasks from the project brief:
  1. REGRESSION      -> predict engagement_rate (continuous)
  2. CLASSIFICATION  -> predict performance_bucket_label (high/low performance)
  3. CLUSTERING      -> K-Means / DBSCAN to discover content-type groupings
                         and unusual high-performing posts (outliers)

Expected input columns:
post_id, account_id, account_type, follower_count, media_type,
content_category, traffic_source, has_call_to_action, post_datetime,
post_date, post_hour, day_of_week, likes, comments, shares, saves,
reach, impressions, engagement_rate, followers_gained, caption_length,
hashtags_count, performance_bucket_label

LEAKAGE NOTE:
likes, comments, shares, saves, reach, impressions, followers_gained
are all post-hoc outcomes (either components of engagement_rate or only
known after the post goes live). They are excluded from every model's
input features (X) below. performance_bucket_label is excluded from
regression/clustering features since it's derived from engagement_rate,
but it IS the classification target.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    silhouette_score,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


RANDOM_STATE = 42

# Columns that leak the target (post-hoc outcomes) — never used as features
LEAKAGE_COLS = [
    "likes", "comments", "shares", "saves",   # raw components of engagement_rate
    "reach", "impressions",                    # denominator of engagement_rate
    "followers_gained",                        # only known after the post's window
    "post_id",                                 # identifier, not predictive
]

CATEGORICAL_COLS = ["account_type", "media_type", "content_category", "traffic_source"]

# With ~20 accounts, account_id can be used as a low-cardinality feature.
# Set False if you want the model to generalize to accounts it hasn't seen.
USE_ACCOUNT_ID_AS_FEATURE = True


# ---------------------------------------------------------------------
# 1. Load data — .env aware (mirrors notebook: COLAB_CSV_PATH / LOCAL_CSV_PATH)
# ---------------------------------------------------------------------
def load_data(path: str = None) -> pd.DataFrame:
    if path is None:
        # .env support like notebook: COLAB_CSV_PATH / LOCAL_CSV_PATH fallback
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        colab_path = os.getenv("COLAB_CSV_PATH", "/content/drive/MyDrive/Test 3/Instagram_Analytics.csv")
        local_path = os.getenv("LOCAL_CSV_PATH", "Instagram_Analytics.csv")
        path = colab_path if os.path.exists(colab_path) else local_path
    df = pd.read_csv(path)
    df["post_datetime"] = pd.to_datetime(df["post_datetime"])
    df["post_date"] = pd.to_datetime(df["post_date"])
    print(f"Loaded {len(df)} rows from: {path}")
    return df


# ---------------------------------------------------------------------
# 2. Feature engineering (shared) — EDA-driven: log+clip both for heavy tails
#    Heavy tails seen in notebook Cell 5/11: likes 199→10632, reach 4913→73339,
#    engagement_rate 0.04→0.27. Apply log1p + 99th clip as user chose "both".
# ---------------------------------------------------------------------
def _clip_99(series: pd.Series) -> pd.Series:
    cap = series.quantile(0.99)
    return series.clip(upper=cap)


def engineer_features(df: pd.DataFrame, fit_train_mean: float = None) -> pd.DataFrame:
    """
    fit_train_mean: mean engagement_rate from train split only — avoids leaking
                    test distribution into train (fixes old global_mean bug).
                    If None, uses df mean (for full-df clustering).
    """
    df = df.copy()

    # --- Cyclical encoding of hour ---
    df["hour_sin"] = np.sin(2 * np.pi * df["post_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["post_hour"] / 24)

    # --- Cyclical encoding of day_of_week ---
    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }
    df["dow_num"] = df["day_of_week"].map(dow_map)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow_num"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow_num"] / 7)
    df["is_weekend"] = df["dow_num"].isin([5, 6]).astype(int)

    # --- Date-derived features ---
    df["month"] = df["post_date"].dt.month
    df["day_of_month"] = df["post_date"].dt.day
    df["quarter"] = df["post_date"].dt.quarter

    # --- Follower count: log transform (20 unique static per account, see EDA) ---
    # EDA: follower_count 5824-10739 IQR, 3083-31095 range — heavy per-account spikes.
    # Use both: 99th clip then log1p (user chose "both").
    df["log_follower_count"] = np.log1p(_clip_99(df["follower_count"]))

    # --- Caption / hashtag ratios ---
    df["hashtags_per_100chars"] = df["hashtags_count"] / (df["caption_length"] + 1) * 100

    # --- Caption/hashtags log+clip (weak signal in EDA Cell 11, but keep) ---
    df["log_caption_length"] = np.log1p(_clip_99(df["caption_length"]))
    df["log_hashtags_count"] = np.log1p(_clip_99(df["hashtags_count"]))

    # --- Rolling / historical features (per account, time-ordered) — KEEP as account history ---
    # User chose A keep as account history: shift(1) avoids instant leakage, only prior posts.
    df = df.sort_values(["account_id", "post_datetime"]).reset_index(drop=True)
    grp = df.groupby("account_id")["engagement_rate"]

    df["prev_engagement_rate"] = grp.shift(1)
    df["rolling_avg_engagement_5"] = (
        grp.shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["rolling_avg_engagement_10"] = (
        grp.shift(1).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["account_post_number"] = df.groupby("account_id").cumcount() + 1

    # Use train mean only (fixes old global_mean leakage before time_based_split)
    fill_val = fit_train_mean if fit_train_mean is not None else df["engagement_rate"].mean()
    for col in ["prev_engagement_rate", "rolling_avg_engagement_5", "rolling_avg_engagement_10"]:
        df[col] = df[col].fillna(fill_val)

    # --- Binary viral target helper ---
    df["is_viral"] = (df["performance_bucket_label"] == "viral").astype(int)

    return df


def engineer_features_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit rolling fillna on train mean only, apply to both splits."""
    train_mean = train_df["engagement_rate"].mean()
    train_df = engineer_features(train_df, fit_train_mean=train_mean)
    test_df = engineer_features(test_df, fit_train_mean=train_mean)
    # For test, rolling windows should see trailing train history per account for realism:
    # recompute test rolling using concatenated tail (time-ordered) then slice.
    # Simpler: keep as-is shift(1) within test only — chronologically correct via time_based_split.
    return train_df, test_df


def build_preprocessor(feature_cols: list):
    """ColumnTransformer: OneHot for categoricals (fixes LabelEncoder false ordinality),
       RobustScaler for numeric (handles outliers per EDA Cell 11), fit on train only."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    if USE_ACCOUNT_ID_AS_FEATURE and "account_id" in feature_cols:
        cat_cols.append("account_id")
    num_cols = [c for c in feature_cols if c not in cat_cols]
    # has_call_to_action is binary numeric, keep as num
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", RobustScaler(), num_cols),
        ],
        remainder="drop",
    )
    return preprocessor, cat_cols, num_cols


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split chronologically so rolling features never leak future->past."""
    df = df.sort_values("post_datetime").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def stratified_time_split(df: pd.DataFrame, test_size: float = 0.2):
    """Fallback stratified split for classification (preserves viral 25% in test)."""
    # Use post_datetime order within each stratum for determinism
    train_parts, test_parts = [], []
    for _, grp in df.groupby("performance_bucket_label"):
        grp = grp.sort_values("post_datetime")
        n_test = max(1, int(len(grp) * test_size))
        train_parts.append(grp.iloc[:-n_test])
        test_parts.append(grp.iloc[-n_test:])
    train_df = pd.concat(train_parts).sort_values("post_datetime").reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values("post_datetime").reset_index(drop=True)
    return train_df, test_df


def get_base_feature_cols(df: pd.DataFrame, exclude_extra=None, include_history: bool = True) -> list:
    exclude = set(LEAKAGE_COLS + [
        "post_datetime", "post_date", "post_hour", "day_of_week", "dow_num",
        "engagement_rate", "performance_bucket_label", "follower_count", "is_viral",
    ])
    if not USE_ACCOUNT_ID_AS_FEATURE:
        exclude.add("account_id")
    if not include_history:
        # If user later wants pure pre-publish without history, drop rolling cols
        exclude.update(["prev_engagement_rate", "rolling_avg_engagement_5", "rolling_avg_engagement_10"])
    if exclude_extra:
        exclude.update(exclude_extra)
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------
# 3. TASK 1 — REGRESSION: predict engagement_rate
#    Stronger model: XGBoost + CatBoost ensemble (vs LGBM baseline)
#    EDA-driven: heavy tails clip+log already in engineer_features
# ---------------------------------------------------------------------
def run_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    print("\n" + "=" * 60)
    print("TASK 1: REGRESSION — predicting engagement_rate (Stronger: XGBoost+CatBoost)")
    print("=" * 60)

    tr_df, val_df = train_test_split(train_df, test_size=0.15, random_state=RANDOM_STATE, shuffle=False)

    preprocessor, _, _ = build_preprocessor(feature_cols)
    X_tr = preprocessor.fit_transform(tr_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])
    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_tr.shape[1])]

    y_tr, y_val = tr_df["engagement_rate"], val_df["engagement_rate"]
    y_test = test_df["engagement_rate"]

    # --- Stronger: XGBoost Regressor (tuned: deeper, regularization) ---
    xgb_reg = xgb.XGBRegressor(
        n_estimators=1000, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, early_stopping_rounds=50, n_jobs=-1,
    )
    xgb_reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    preds_xgb = xgb_reg.predict(X_test)

    # --- Stronger: CatBoost (if available) or HistGradientBoosting fallback ---
    if HAS_CATBOOST:
        cat_reg = CatBoostRegressor(
            iterations=1000, depth=8, learning_rate=0.03, l2_leaf_reg=3,
            random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50,
        )
        cat_reg.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        preds_cat = cat_reg.predict(X_test)
    else:
        hgb_reg = HistGradientBoostingRegressor(max_iter=1000, max_depth=8, learning_rate=0.03, random_state=RANDOM_STATE, early_stopping=True)
        hgb_reg.fit(X_tr, y_tr)
        preds_cat = hgb_reg.predict(X_test)

    # Ensemble average (stronger than single LGBM)
    preds = 0.6 * preds_xgb + 0.4 * preds_cat
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"[XGBoost] MAE: {mean_absolute_error(y_test, preds_xgb):.4f} | R^2: {r2_score(y_test, preds_xgb):.4f} | best: {xgb_reg.best_iteration}")
    if HAS_CATBOOST:
        print(f"[CatBoost] MAE: {mean_absolute_error(y_test, preds_cat):.4f} | R^2: {r2_score(y_test, preds_cat):.4f}")
    print(f"[Ensemble 0.6 XGB + 0.4 {'CatBoost' if HAS_CATBOOST else 'HGB'}] MAE: {mae:.4f} | RMSE: {rmse:.4f} | R^2: {r2:.4f}")

    # Importance from XGBoost gain
    try:
        importance = pd.Series(xgb_reg.feature_importances_, index=feat_names).sort_values(ascending=False)
    except Exception:
        importance = pd.Series(np.abs(preds_xgb[:len(feat_names)]), index=feat_names).sort_values(ascending=False)
    print("\nTop 10 features (XGBoost gain):")
    print(importance.head(10))

    return (xgb_reg, cat_reg if HAS_CATBOOST else hgb_reg), preprocessor, importance


# ---------------------------------------------------------------------
# 4. TASK 2 — CLASSIFICATION: Stronger XGBoost + CatBoost (vs LGBM baseline)
#    User asked stronger model — now XGBoost depth 8 + CatBoost ordered boosting
# ---------------------------------------------------------------------
def run_classification(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    print("\n" + "=" * 60)
    print("TASK 2a: CLASSIFICATION — 4-class (Stronger: XGBoost+CatBoost)")
    print("=" * 60)

    label_encoder = LabelEncoder()
    y_train_4 = label_encoder.fit_transform(train_df["performance_bucket_label"].astype(str))
    y_test_4 = label_encoder.transform(test_df["performance_bucket_label"].astype(str))

    tr_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_4)
    preprocessor, _, _ = build_preprocessor(feature_cols)
    X_tr = preprocessor.fit_transform(train_df.iloc[tr_idx][feature_cols])
    X_val = preprocessor.transform(train_df.iloc[val_idx][feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])
    y_tr, y_val = y_train_4[tr_idx], y_train_4[val_idx]

    # XGBoost multi-class (stronger: max_depth 8, reg, larger estimators)
    xgb_4 = xgb.XGBClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, early_stopping_rounds=50,
    )
    xgb_4.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    preds_4 = xgb_4.predict(X_test)
    print(f"[XGBoost] Accuracy: {accuracy_score(y_test_4, preds_4):.4f} | F1 weighted: {f1_score(y_test_4, preds_4, average='weighted'):.4f} | F1 macro: {f1_score(y_test_4, preds_4, average='macro'):.4f} | best: {xgb_4.best_iteration}")

    # CatBoost 4-class as second stronger model
    if HAS_CATBOOST:
        cat_4 = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.03, l2_leaf_reg=3, random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50, auto_class_weights="Balanced")
        cat_4.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        preds_cat4 = cat_4.predict(X_test)
        print(f"[CatBoost] Accuracy: {accuracy_score(y_test_4, preds_cat4):.4f} | F1 weighted: {f1_score(y_test_4, preds_cat4, average='weighted'):.4f}")
    else:
        cat_4 = None

    print("\nClassification report (XGBoost 4-class):")
    print(classification_report(y_test_4, preds_4, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_4, preds_4))
    print(f"best_iter: {xgb_4.best_iteration}")

    print("\n" + "=" * 60)
    print("TASK 2b: CLASSIFICATION — binary is_viral (Stronger: XGBoost+CatBoost)")
    print("=" * 60)
    y_train_b = train_df["is_viral"].astype(int).values
    y_test_b = test_df["is_viral"].astype(int).values
    tr_idx_b, val_idx_b = train_test_split(np.arange(len(train_df)), test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_b)
    preprocessor_b, _, _ = build_preprocessor(feature_cols)
    X_tr_b = preprocessor_b.fit_transform(train_df.iloc[tr_idx_b][feature_cols])
    X_val_b = preprocessor_b.transform(train_df.iloc[val_idx_b][feature_cols])
    X_test_b = preprocessor_b.transform(test_df[feature_cols])
    y_tr_b, y_val_b = y_train_b[tr_idx_b], y_train_b[val_idx_b]

    # scale_pos_weight for 75/25 imbalance
    neg, pos = (y_tr_b == 0).sum(), (y_tr_b == 1).sum()
    spw = neg / max(pos, 1)

    xgb_b = xgb.XGBClassifier(
        n_estimators=1000, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1, early_stopping_rounds=50,
    )
    xgb_b.fit(X_tr_b, y_tr_b, eval_set=[(X_val_b, y_val_b)], verbose=False)
    preds_b = xgb_b.predict(X_test_b)
    proba_b = xgb_b.predict_proba(X_test_b)[:, 1]
    print(f"[XGBoost] Accuracy: {accuracy_score(y_test_b, preds_b):.4f} | Precision viral: {precision_score(y_test_b, preds_b, zero_division=0):.4f} | Recall viral: {recall_score(y_test_b, preds_b, zero_division=0):.4f} | F1 viral: {f1_score(y_test_b, preds_b, zero_division=0):.4f} | ROC-AUC: {roc_auc_score(y_test_b, proba_b):.4f} | PR-AUC: {average_precision_score(y_test_b, proba_b):.4f} | best: {xgb_b.best_iteration}")

    if HAS_CATBOOST:
        cat_b = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.03, l2_leaf_reg=3, random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50, auto_class_weights="Balanced")
        cat_b.fit(X_tr_b, y_tr_b, eval_set=(X_val_b, y_val_b), verbose=False)
        preds_catb = cat_b.predict(X_test_b)
        proba_catb = cat_b.predict_proba(X_test_b)[:, 1]
        print(f"[CatBoost] Accuracy: {accuracy_score(y_test_b, preds_catb):.4f} | F1 viral: {f1_score(y_test_b, preds_catb, zero_division=0):.4f} | ROC-AUC: {roc_auc_score(y_test_b, proba_catb):.4f} | PR-AUC: {average_precision_score(y_test_b, proba_catb):.4f}")
        # Ensemble
        ensemble_proba = 0.6 * proba_b + 0.4 * proba_catb
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        print(f"[Ensemble 0.6 XGB + 0.4 CatBoost] F1 viral: {f1_score(y_test_b, ensemble_pred, zero_division=0):.4f} | ROC-AUC: {roc_auc_score(y_test_b, ensemble_proba):.4f}")
    else:
        cat_b = None

    print("\nClassification report (XGBoost binary viral):")
    print(classification_report(y_test_b, preds_b, target_names=["not_viral", "viral"]))
    print("Confusion matrix (viral):")
    print(confusion_matrix(y_test_b, preds_b))
    print(f"best_iter: {xgb_b.best_iteration}")

    return (xgb_4, preprocessor, label_encoder, cat_4), (xgb_b, preprocessor_b, cat_b)


# ---------------------------------------------------------------------
# 5. TASK 3 — CLUSTERING: K-Means & DBSCAN (EDA-driven fixes)
#    EDA Cell 11/12: heavy tails + mixed categoricals need OneHot+RobustScaler,
#    LabelEncoder broke distance, eps=1.2 gave 100% outliers. Sweep k and tune eps.
# ---------------------------------------------------------------------
def run_clustering(df: pd.DataFrame, feature_cols: list, k: int = 5, dbscan_eps: float = None, dbscan_min_samples: int = 15):
    print("\n" + "=" * 60)
    print("TASK 3: CLUSTERING — content-type groupings & unusual posts")
    print("=" * 60)

    # Use OneHot+RobustScaler built on feature_cols (fixes LabelEncoder ordinality)
    # Drop account_id noise, keep traffic_source (user chose keep) with OneHot
    cluster_feature_cols = [c for c in feature_cols if c not in ("account_id",)]
    preprocessor, _, _ = build_preprocessor(cluster_feature_cols)
    # Fit on content features only; engagement_rate excluded from distance, used only to label
    X_scaled = preprocessor.fit_transform(df[cluster_feature_cols])
    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_scaled.shape[1])]
    print(f"Clustering on {X_scaled.shape[1]} OneHot+scaled features (from {len(cluster_feature_cols)} raw). Example: {list(feat_names)[:6]}")

    # --- K-Means sweep 3-8 (EDA showed silhouette 0.07 at k=5 — seek better) ---
    best_k, best_score, best_kmeans = k, -1, None
    for cand_k in [3, 4, 5, 6, 8]:
        km = KMeans(n_clusters=cand_k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        # silhouette undefined for single cluster, skip
        try:
            s = silhouette_score(X_scaled, labels)
        except Exception:
            s = -1
        print(f"  k={cand_k} silhouette={s:.4f}")
        if s > best_score:
            best_score, best_k, best_kmeans = s, cand_k, km
    kmeans = best_kmeans
    kmeans_labels = kmeans.labels_
    df["kmeans_cluster"] = kmeans_labels
    print(f"\nChosen K-Means k={best_k} silhouette={best_score:.4f}")

    print("\nMean engagement_rate per K-Means cluster (sorted):")
    cluster_summary = (
        df.groupby("kmeans_cluster")["engagement_rate"]
        .agg(["mean", "count", "std"])
        .sort_values("mean", ascending=False)
    )
    print(cluster_summary)

    # --- DBSCAN tune eps via k-distance (fixes 100% outliers at 1.2) ---
    if dbscan_eps is None:
        # 5th nearest neighbor distance heuristic
        nn = NearestNeighbors(n_neighbors=5).fit(X_scaled)
        dists, _ = nn.kneighbors(X_scaled)
        k_dist = np.sort(dists[:, -1])
        # percentile heuristic: eps at 90th percentile of k-distance
        dbscan_eps = float(np.percentile(k_dist, 90))
        print(f"Auto-tuned DBSCAN eps={dbscan_eps:.3f} (90th pct k-dist), min_samples={dbscan_min_samples}")
        # Also report k-dist stats for transparency
        print(f"k-dist 50th={np.percentile(k_dist,50):.3f} 75th={np.percentile(k_dist,75):.3f} 95th={np.percentile(k_dist,95):.3f}")

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    df["dbscan_cluster"] = dbscan_labels

    n_outliers = (dbscan_labels == -1).sum()
    n_clusters_found = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"\nDBSCAN found {n_clusters_found} clusters and {n_outliers} outlier posts "
          f"({n_outliers / len(df) * 100:.1f}% of data).")

    # Outliers with above-average engagement = "unusual high-performing posts"
    avg_engagement = df["engagement_rate"].mean()
    unusual_high_performers = df[
        (df["dbscan_cluster"] == -1) & (df["engagement_rate"] > avg_engagement)
    ]
    print(f"Unusual HIGH-performing outlier posts: {len(unusual_high_performers)}")
    if len(unusual_high_performers) > 0:
        print(unusual_high_performers[["post_id", "account_type", "media_type", "content_category", "engagement_rate", "follower_count", "is_viral"]].head(5).to_string())

    return kmeans, dbscan, preprocessor, df


# ---------------------------------------------------------------------
# 6. Main — respects keep/history + both outlier fixes
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="CSV path (default: .env COLAB_CSV_PATH/LOCAL_CSV_PATH)")
    args = parser.parse_args()
    DATA_PATH = args.data

    raw_df = load_data(DATA_PATH)
    # Split raw first, then engineer with train-mean fillna (fixes leakage)
    raw_train, raw_test = time_based_split(raw_df, test_size=0.2)
    train_df, test_df = engineer_features_train_test(raw_train, raw_test)
    # For clustering we need full df engineered with train mean (so history consistent)
    full_df = pd.concat([train_df, test_df]).sort_values("post_datetime").reset_index(drop=True)
    # Also need full engineered for clustering counts (re-engineer with train mean for full)
    # full_df already engineered via the two splits

    feature_cols = get_base_feature_cols(full_df)
    print(f"Using {len(feature_cols)} features (keep history, keep traffic_source, clip+log both):\n{feature_cols}\n")

    # Task 1: Regression
    reg_model, reg_preproc, reg_importance = run_regression(train_df, test_df, feature_cols)

    # Task 2: Classification — stronger XGBoost+CatBoost (both 4-class keep + binary viral)
    (clf_model_4, clf_preproc_4, label_encoder, clf_cat_4), (clf_model_b, clf_preproc_b, clf_cat_b) = run_classification(train_df, test_df, feature_cols)

    # Task 3: Clustering (uses full dataset, not train/test split)
    kmeans_model, dbscan_model, cluster_preproc, df_with_clusters = run_clustering(
        full_df, feature_cols, k=5
    )

    print("\nDone. Stronger Models: reg (XGB+CatBoost ensemble), clf_4 (XGB+CatBoost 4-class), clf_b (XGB+CatBoost viral), kmeans, dbscan")
