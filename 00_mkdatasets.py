from pathlib import Path
import pandas as pd
from src.utils.logger import KABUFormatter, get_logger
logger = get_logger()


INPUT_DIR = Path("filtered")
PATTERN = "bb_rsi_valley_summary_*_3rd_bid_4th_ASK_SETS.csv"

OUTPUT_DIR = Path("datasets")
OUTPUT_FILE = "brv_datasets.csv"

DROP_COLUMNS = [
    "Exit_Dist_Pct",
    "Upper_Shadow_Pct",
    "Filter_Group_Id",
    "Filter_Target_Index",
    "Group_Id",
]


def build_feature(group: pd.DataFrame) -> dict | None:
    stages = {
        stage: group[group["Stage"] == stage]
        for stage in (
            "1st_RSI",
            "2nd_BB",
            "3rd_VALLEY",
            "4th_EXIT",
        )
    }

    valley = stages["3rd_VALLEY"]
    exit_ = stages["4th_EXIT"]

    if valley.empty or exit_.empty:
        return None

    buy_price = valley["Detect_Price"].iloc[0]
    sell_price = exit_["Detect_Price"].iloc[0]
    profit_pct = (sell_price - buy_price) / buy_price * 100

    feature = valley.iloc[0].to_dict()

    for stage_df in stages.values():
        if stage_df.empty:
            continue

        for col, value in stage_df.iloc[0].items():
            if pd.isna(feature.get(col)):
                feature[col] = value

    feature.update(
        {
            "Group_Id": group["Filter_Group_Id"].iloc[0],
            "Buy_Price": buy_price,
            "Profit_Pct": profit_pct,
            "Target_Profit": int(profit_pct >= 5),
        }
    )

    return feature


def main() -> None:
    logger.info("CSVファイルを検索します")

    files = sorted(INPUT_DIR.glob(PATTERN))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {INPUT_DIR.resolve()}"
        )

    logger.info("CSVファイル数: %d", len(files))

    df = pd.concat(
        (pd.read_csv(file) for file in files),
        ignore_index=True,
    )

    logger.info("データ総行数: %d", len(df))
    logger.info("特徴量作成開始")

    features = []

    for _, group in df.groupby(
        ["Ticker", "Filter_Group_Id"],
        sort=False,
    ):
        feature = build_feature(group)

        if feature:
            features.append(feature)

    features_df = (
        pd.DataFrame(features)
        .dropna(subset=["Target_Profit"])
        .dropna(axis=1, how="all")
        .drop_duplicates(subset=["Ticker", "Detect_Date"])
        .drop(columns=DROP_COLUMNS, errors="ignore")
    )

    logger.info("最終データ数: %d", len(features_df))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(
        OUTPUT_DIR / OUTPUT_FILE,
        index=False,
    )

    logger.info("CSV保存完了")
    logger.info("データセットの先頭5行:")
    logger.info(features_df.head())


if __name__ == "__main__":
    main()