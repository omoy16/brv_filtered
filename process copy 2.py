import pandas as pd


def analyze_pair(df, feature1, feature2, bins=3):
    work = df.copy()

    work[f"{feature1}_Bin"] = pd.qcut(
        work[feature1],
        q=bins,
        duplicates="drop"
    )

    work[f"{feature2}_Bin"] = pd.qcut(
        work[feature2],
        q=bins,
        duplicates="drop"
    )

    winrate = pd.pivot_table(
        work,
        values="Target_Profit",
        index=f"{feature1}_Bin",
        columns=f"{feature2}_Bin",
        aggfunc="mean"
    )

    count = pd.pivot_table(
        work,
        values="Target_Profit",
        index=f"{feature1}_Bin",
        columns=f"{feature2}_Bin",
        aggfunc="count"
    )

    print("=" * 80)
    print(f"{feature1} × {feature2}")
    print("\n【WinRate】")
    print(winrate)

    print("\n【Count】")
    print(count)

    return winrate, count