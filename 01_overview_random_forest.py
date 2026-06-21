# %%
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# %% [markdown]
#### データセットの読み込みと基本確認
# %%
features_df = pd.read_csv("./datasets/brv_datasets.csv")

missing = features_df.isnull().sum()
missing = missing[missing > 0]

logger.info("=== Dataset Summary ===")
logger.info("shape: %s", features_df.shape)
logger.info("duplicates: %d", features_df.duplicated().sum())

if not missing.empty:
    logger.info("missing values:\n%s", missing.to_dict())

logger.info("head:\n%s", features_df.head())

# %%
# BB欠損行だけの勝率を見る
bb_null = features_df[features_df["BB_Width"].isna()]
bb_exist = features_df[features_df["BB_Width"].notna()]

print("\n=== BB_Width 欠損の有無による勝率比較 ===")
print("欠損件数", len(bb_null))
print("非欠損件数", len(bb_exist))
print("欠損勝率")
print(bb_null["Target_Profit"].mean())
print("非欠損勝率")
print(bb_exist["Target_Profit"].mean())

# %% [markdown]
#### ランダムフォレストモデルの学習と評価
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# %%
# Feature Selection
FEATURES_TO_KEEP = [
    "RSI",
    "Prev_RSI",
    "RSI_Diff",
    "RSI_Drop_Pct",
    "RSI_Depth",
    "RSI_Down_Streak",
    "RSI_Rank20",
    "RSI_Zscore20",
    "BB_Days",
    "BB_Close_Ratio",
    "BB_Touch_Ratio",
    "BB_Width",
    "BB_Position",
    "BB_Width_Zscore20",
    "Lower_Shadow_Pct",
    "Volume_Ratio",
    "Valley_Body_Pct",
    "Volume_Zscore20",
]

missing_cols = set(FEATURES_TO_KEEP) - set(features_df.columns)
if missing_cols:
    raise ValueError(f"列不足: {missing_cols}")
if "Target_Profit" not in features_df.columns:
    raise ValueError("Target_Profit が存在しません")

# %%
# Data Preparation
X = features_df[FEATURES_TO_KEEP]
y = features_df["Target_Profit"]

# NaNが含まれると学習できないため、0で補完
X = X.fillna(0)

print("=== 学習に使用する特徴量 ===")
print(X.columns.tolist())
print(f"=== 特徴量の形状 ===\n {X.shape}")

# %%
# Target Distribution
print("=== 学習前のターゲット分布 ===")
print(y.value_counts())
print(y.value_counts(normalize=True))

# %%
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n=== 分割後のサンプル数 ===\nTrain: {len(X_train)}, Test: {len(X_test)}")

print("\n=== 分割後のクラス分布 (Train) ===")
print(y_train.value_counts(normalize=True))
print("=== 分割後のクラス分布 (Test) ===")
print(y_test.value_counts(normalize=True))

# %%
# Train Model
print("\n=== ランダムフォレストモデルの学習 ===")
rf_clf = RandomForestClassifier(
    n_estimators=300, max_depth=5, random_state=42, class_weight="balanced"
)
rf_clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, rf_clf.predict(X_train))
test_acc = accuracy_score(y_test, rf_clf.predict(X_test))
print(f"\n訓練精度: {train_acc:.3f}")
print(f"テスト精度: {test_acc:.3f}")
print(f"精度ギャップ (Overfitting check): {train_acc - test_acc:.3f}")

# %%
# Feature Importance
print("\n=== 特徴量の重要度 ===")
importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": rf_clf.feature_importances_}
).sort_values("importance", ascending=False)
print(importance_df)

# %%
# Permutation Importance
print("\n=== Permutation Importance (Test Data) ===")
perm_importance = permutation_importance(
    rf_clf, X_test, y_test, n_repeats=10, random_state=42
)
perm_importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": perm_importance.importances_mean,
        "std": perm_importance.importances_std,
    }
).sort_values("importance", ascending=False)
print(perm_importance_df)

# %%
# Model Evaluation
y_pred = rf_clf.predict(X_test)
print("\n=== 予測クラス分布 (Test) ===")
print(pd.Series(y_pred).value_counts())

print("\n=== 混同行列 (Test) ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== 最終評価 (テストデータ) ===")
print(classification_report(y_test, y_pred))

# %%
# AUC Evaluation
y_proba = rf_clf.predict_proba(X_test)[:, 1]
print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"PR-AUC (Average Precision): {average_precision_score(y_test, y_proba):.3f}")

# %%
# Calibration
print("\n=== 予測確率ごとの勝率 (Calibration) ===")
prob_summary = (
    pd.DataFrame({"Predicted_Prob": y_proba.round(2), "Actual_Win": y_test.values})
    .groupby("Predicted_Prob")
    .agg(Actual_Win_Rate=("Actual_Win", "mean"), Sample_Count=("Actual_Win", "count"))
    .sort_index(ascending=False)
)
print(prob_summary)

# %%
# Feature Analysis (Quantiles)
for col in [
    "BB_Position",
    "BB_Width",
    "BB_Close_Ratio",
    "BB_Touch_Ratio",
]:
    print(f"\n=== {col} ===")
    tmp = features_df.groupby(
        observed=True, by=pd.qcut(features_df[col], 5, duplicates="drop")
    )["Target_Profit"].mean()
    print(tmp)

# %%
# Win Rate Analysis by Probability
proba = rf_clf.predict_proba(X_test)[:, 1]

result = pd.DataFrame(
    {
        "Pred_Prob": proba,
        "Actual": y_test.values,
    },
    index=X_test.index,
)

result = result.sort_values("Pred_Prob", ascending=False)
print(result.head())

# %%
# Top % Win Rate
for pct in [0.1, 0.2, 0.3, 0.5]:
    n = int(len(result) * pct)
    subset = result.head(n)
    print(f"上位{int(pct*100)}% 件数={n} 勝率={subset['Actual'].mean():.3f}")

# %%
# Top 10% Feature Analysis
top10_idx = result.head(int(len(result) * 0.1)).index
top10_features = X_test.loc[top10_idx]

print("\n=== 上位10%の選別個体分析 (vs テストデータ全体) ===")
target_cols = ["Volume_Ratio", "BB_Width", "Lower_Shadow_Pct"]

for col in target_cols:
    top_mean = top10_features[col].mean()
    all_mean = X_test[col].mean()
    print(
        f"{col:<18} | 上位10%: {top_mean:>8.4f} | 全体: {all_mean:>8.4f} | 差: {top_mean - all_mean:>+8.4f}"
    )

# %%
# Top 10% Statistics
for col in [
    "Volume_Ratio",
    "BB_Width",
    "Lower_Shadow_Pct",
]:
    print(top10_features[col].describe())
