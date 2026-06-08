import pandas as pd
from pathlib import Path

# ### 1. データの読み込みと基本確認
# 指定パターンのCSVをすべて読み込んで結合
files = Path(".").glob("bb_rsi_valley_summary_*_3rd_bid_4th_ASK_SETS.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print(df.head())

# ### 2. 特徴量の統合 (Feature Engineering)
features = []

# TickerとGroup_Idの両方でグループ化（複数ファイル結合時のID重複対策）
for (ticker, group_id), group in df.groupby(["Ticker", "Filter_Group_Id"]):
    # 必要なステージを抽出
    stages = {s: group[group["Stage"] == s] for s in ["1st_RSI", "2nd_BB", "3rd_VALLEY", "4th_EXIT"]}

    # Valley(買い)とExit(売り)の両方が揃っている場合のみ処理
    if not stages["3rd_VALLEY"].empty and not stages["4th_EXIT"].empty:
        buy_price = stages["3rd_VALLEY"]["Detect_Price"].values[0]
        sell_price = stages["4th_EXIT"]["Detect_Price"].values[0]
        profit_pct = ((sell_price - buy_price) / buy_price) * 100

        # 基本となる行（Valley）を作成
        feature_row = stages["3rd_VALLEY"].iloc[0].to_dict()

        # 他のステージにある欠損値を補完（RSIやBBの指標を統合）
        for s_name, s_df in stages.items():
            if not s_df.empty:
                for col, val in s_df.iloc[0].to_dict().items():
                    if pd.isna(feature_row.get(col)):
                        feature_row[col] = val

        feature_row["Group_Id"] = group_id
        feature_row["Buy_Price"] = buy_price
        feature_row["Target_Profit"] = 1 if profit_pct > 0 else 0
        features.append(feature_row)

features_df = pd.DataFrame(features).dropna(subset=["Target_Profit"]).dropna(axis=1, how="all")

# 重複排除: 同銘柄かつDetect_Dateが同じものを削除
initial_count = len(features_df)
features_df = features_df.drop_duplicates(subset=['Ticker', 'Detect_Date'])
final_count = len(features_df)

print(f"元のサンプル数: {initial_count}")
print(f"重複排除後のサンプル数: {final_count} (削除数: {initial_count - final_count})")
print(features_df.head())
features_df.to_csv("brv_4th_ASK_SETS_features.csv", index=False)


# ### 3. 決定木モデルの学習と評価
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.metrics import classification_report

# 数値データの抽出
X = features_df.select_dtypes(include="number").drop("Target_Profit", axis=1)

# 学習前に特徴量の確認
print("=== 学習に使用する特徴量 ===")
print(X.columns.tolist())

# NaNが含まれると学習できないため、0で補完（必要に応じて平均値などに変更）
X = X.fillna(0)
y = features_df["Target_Profit"]

# 分割と学習
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 不均衡データに対応するため class_weight='balanced' を追加
tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42, class_weight='balanced')
tree_clf.fit(X_train, y_train)

print(f"訓練精度: {accuracy_score(y_train, tree_clf.predict(X_train)):.3f}")
print(f"テスト精度: {accuracy_score(y_test, tree_clf.predict(X_test)):.3f}")

# ルールの表示
print("\n=== AIが導き出したルール ===")
print(export_text(tree_clf, feature_names=list(X.columns)))

print("\n=== 分類レポート ===")
print(classification_report(y_test, tree_clf.predict(X_test)))
