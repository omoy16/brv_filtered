import pandas as pd

# ### データの読み込みと基本確認
features_df = pd.read_csv("./datasets/brv_datasets.csv")

print("=== データの基本情報 ===")
print(features_df.info())
print("=== データの形状 ===")
print(features_df.shape)
print("=== データの先頭行 ===")
print(features_df.head())
print("\n=== 欠損値 ===")
print(features_df.isnull().sum())

print("\n=== 重複行のチェック ===")
print("duplicates:", features_df.duplicated().sum())

# ② BB欠損行だけの勝率を見る
bb_null = features_df[features_df["BB_Width"].isna()]
bb_exist = features_df[features_df["BB_Width"].notna()]

print("\n=== BB_Width 欠損の有無による勝率比較 ===")
print("欠損件数", len(bb_null))
print("非欠損件数", len(bb_exist))
print("欠損勝率")
print(bb_null["Target_Profit"].mean())
print("非欠損勝率")
print(bb_exist["Target_Profit"].mean())


# ### 決定木モデルの学習と評価
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.tree import export_text
from sklearn.inspection import permutation_importance

# 特定の特徴量のみに絞り込む
FEATURES_TO_KEEP = [
    "RSI", "Prev_RSI", "RSI_Diff", "RSI_Drop_Pct", "RSI_Depth",
    "RSI_Down_Streak", "RSI_Rank20", "RSI_Zscore20", "BB_Days",
    "BB_Close_Ratio", "BB_Touch_Ratio", "BB_Width", "BB_Position",
    "BB_Width_Zscore20", "Lower_Shadow_Pct", "Volume_Ratio",
    "Valley_Body_Pct", "Volume_Zscore20",
]


missing_cols = set(FEATURES_TO_KEEP) - set(features_df.columns)
if missing_cols:
    raise ValueError(
        f"列不足: {missing_cols}"
    )
if "Target_Profit" not in features_df.columns:
    raise ValueError("Target_Profit が存在しません")

# 特徴量とターゲットの抽出（データリーク防止のため明示的に指定）
X = features_df[FEATURES_TO_KEEP]
y = features_df["Target_Profit"]

# 学習前に特徴量の確認
print("=== 学習に使用する特徴量 ===")
print(X.columns.tolist())
print(f"=== 特徴量の形状 ===\n {X.shape}")

# NaNが含まれると学習できないため、0で補完（必要に応じて平均値などに変更）
X = X.fillna(0)

# 学習前の評価
print("=== 学習前のターゲット分布 ===")
print(y.value_counts())
print(y.value_counts(normalize=True)) # 割合も見たい

# データの分割後のサンプル数を確認
print("\n=== 分割後のサンプル数 ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

print("\n=== 分割後のクラス分布 (Train) ===")
print(y_train.value_counts(normalize=True))
print("=== 分割後のクラス分布 (Test) ===")
print(y_test.value_counts(normalize=True))

# 不均衡データに対応するため class_weight='balanced' を追加
tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42, class_weight='balanced')
tree_clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, tree_clf.predict(X_train))
test_acc = accuracy_score(y_test, tree_clf.predict(X_test))
print(f"\n訓練精度: {train_acc:.3f}")
print(f"テスト精度: {test_acc:.3f}")
print(f"精度ギャップ (Overfitting check): {train_acc - test_acc:.3f}")

# 特徴量の重要度を確認
print("\n=== 特徴量の重要度 ===")
importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": tree_clf.feature_importances_
    }
).sort_values("importance", ascending=False)
print(importance_df)

# Permutation Importance (テストデータ) を確認
print("\n=== Permutation Importance (Test Data) ===")
perm_importance = permutation_importance(tree_clf, X_test, y_test, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": perm_importance.importances_mean,
        "std": perm_importance.importances_std
    }
).sort_values("importance", ascending=False)
print(perm_importance_df)

# ルールの表示
print("\n=== AIが導き出したルール ===")
print(export_text(tree_clf, feature_names=list(X.columns)))

y_pred = tree_clf.predict(X_test)
print("\n=== 予測クラス分布 (Test) ===")
print(pd.Series(y_pred).value_counts())

print("\n=== 混同行列 (Test) ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== 最終評価 (テストデータ) ===")
print(classification_report(y_test, y_pred))

# AUC関連の評価
y_proba = tree_clf.predict_proba(X_test)[:, 1]
print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"PR-AUC (Average Precision): {average_precision_score(y_test, y_proba):.3f}")

print("\n=== 予測確率ごとの勝率 (Calibration) ===")
# 決定木はリーフごとに確率が決まるため、そのままグループ化して集計
prob_summary = pd.DataFrame({
    "Predicted_Prob": y_proba,
    "Actual_Win": y_test.values
}).groupby("Predicted_Prob").agg(
    Actual_Win_Rate=("Actual_Win", "mean"),
    Sample_Count=("Actual_Win", "count")
).sort_index(ascending=False) # 確率が高い順に表示
print(prob_summary)

for col in [
    "BB_Position",
    "BB_Width",
    "BB_Close_Ratio",
    "BB_Touch_Ratio",
]:
    print(f"\n=== {col} ===")

    tmp = (
        features_df.groupby(
            observed=True, 
            by=pd.qcut(features_df[col], 5, duplicates="drop")
        )["Target_Profit"]
        .mean()
    )

    print(tmp)



# for col in [
#     "RSI_Diff",
#     "Lower_Shadow_Pct",
#     "BB_Days",
#     "Volume_Ratio",
#     "BB_Close_Ratio"
# ]:
#     print("\n", col)
#     print(
#         features_df.groupby(
#             pd.qcut(features_df[col], 5, duplicates="drop")
#         )["Target_Profit"].mean()
#     )
    
    
# for col in FEATURES_TO_KEEP:
#     print(col)
#     print(features_df[col].describe())
#     print()

# corr = features_df.corr(numeric_only=True)
# print(corr["RSI"].sort_values())
