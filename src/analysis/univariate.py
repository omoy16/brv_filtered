import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class UnivariateResult:
    feature_name: str
    summary_df: pd.DataFrame


@dataclass
class BivariateResult:
    feature1: str
    feature2: str
    winrate_df: pd.DataFrame
    count_df: pd.DataFrame


def analyze_feature(
    df: pd.DataFrame,
    feature: str,
    target: str = "Target_Profit",
    profit_col: str = "Profit_Pct",
    bins: int = 5,
) -> UnivariateResult:
    """
    Analyze a single feature by binning it and calculating target statistics.
    """
    # Use a copy to avoid modifying the original dataframe
    work = df[[feature, target, profit_col]].copy()

    # Binning the feature
    work["Bin"] = pd.qcut(work[feature], q=bins, duplicates="drop")

    # Aggregating statistics
    result_df = work.groupby("Bin", observed=True).agg(
        Count=(target, "count"),
        WinRate=(target, "mean"),
        AvgProfit=(profit_col, "mean"),
        MedianProfit=(profit_col, "median"),
    )

    return UnivariateResult(feature_name=feature, summary_df=result_df)


def analyze_features(
    df: pd.DataFrame,
    features: list[str],
    target: str = "Target_Profit",
    profit_col: str = "Profit_Pct",
    bins: int = 5,
) -> list[UnivariateResult]:
    """
    Analyze multiple features and return a list of UnivariateResult objects.
    """
    return [
        analyze_feature(df, f, target=target, profit_col=profit_col, bins=bins)
        for f in features
    ]


def export_univariate_results(
    results: list[UnivariateResult], output_dir: str = "outputs/univariate"
) -> list[str]:
    """
    Export UnivariateResult objects to CSV files in the specified directory.
    Returns a list of saved file paths.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []
    for res in results:
        file_path = os.path.join(output_dir, f"{res.feature_name}.csv")
        res.summary_df.to_csv(file_path)
        saved_paths.append(file_path)

    return saved_paths


def analyze_pair(
    df: pd.DataFrame,
    feature1: str,
    feature2: str,
    target: str = "Target_Profit",
    bins: int = 3,
) -> BivariateResult:
    """
    Analyze the interaction between two features using a pivot table.
    """
    work = df.copy()

    work[f"{feature1}_Bin"] = pd.qcut(work[feature1], q=bins, duplicates="drop")
    work[f"{feature2}_Bin"] = pd.qcut(work[feature2], q=bins, duplicates="drop")

    winrate = pd.pivot_table(
        work,
        values=target,
        index=f"{feature1}_Bin",
        columns=f"{feature2}_Bin",
        aggfunc="mean",
        observed=False,
    )

    count = pd.pivot_table(
        work,
        values=target,
        index=f"{feature1}_Bin",
        columns=f"{feature2}_Bin",
        aggfunc="count",
        observed=False,
    )

    return BivariateResult(
        feature1=feature1,
        feature2=feature2,
        winrate_df=winrate,
        count_df=count,
    )
