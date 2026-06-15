import pandas as pd

datasets_path = r'C:\Users\omoy1\Desktop\brv_filtered\brv_datasets.csv'
features_df = pd.read_csv(datasets_path)
features_df.head()

for col in features_df.columns:
    print(
        f"{col:<25}",
        features_df[col].dtype
    )
    
feature_cols = features_df.select_dtypes(include=["number"]).columns.tolist()

exclude_cols = [
    "Target_Profit",
    "Profit_Pct",
    "Buy_Price",
    "Filter_Relative_Pos",
    "Detect_Price",
    "Last_Price",
]
feature_cols = [col for col in features_df.columns if col in feature_cols and col not in exclude_cols]
    
import sweetviz as sv

report = sv.analyze(features_df[feature_cols + ["Target_Profit"]])
report.show_html()