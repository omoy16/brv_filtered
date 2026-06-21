## Plan: Univariate分析のライブラリ化とNotebook整理

Notebook内の分析ロジックを `KABUCALIB/src/analysis/univariate.py` に切り出し、再利用可能な形式にします。これにより、Notebookは「設定と可視化」のみに特化させ、分析の再現性と保守性を向上させます。

**Steps**

### Phase 1: 分析コアロジックの切り出し
1. `KABUCALIB/src/analysis/` ディレクトリを作成する。
2. `KABUCALIB/src/analysis/univariate.py` を作成し、以下の関数を実装する。
    - `analyze_feature(df, feature, target="Target_Profit", profit_col="Profit_Pct", bins=5)`: 
        - `pd.qcut` によるビン分け、`groupby` による `Count`, `WinRate`, `AvgProfit`, `MedianProfit` の算出。
    - `analyze_pair(df, feature1, feature2, target="Target_Profit", bins=3)`: 
        - 2変数間のクロス集計（WinRateとCountのピボットテーブル作成）。
3. Notebook `univariate_analysis copy.ipynb` でこれらの関数をインポートし、既存のインライン関数を置き換えて結果が一致することを検証する。

### Phase 2: 一括分析と出力の自動化
1. `univariate.py` に `analyze_features(df, features, ...)` を追加し、複数特徴量をループして結果を辞書形式で集約する機能を実装する。
2. 分析結果を `KABUCALIB/outputs/univariate/` にCSVとして保存する `export_univariate_results` 関数を実装する。
3. Notebook側で `FEATURES` リストを定義し、一括分析 $\rightarrow$ 保存の流れを構築する。

### Phase 3: Notebookの整理と最適化
1. `univariate_analysis copy.ipynb` から冗長な計算コードを削除し、`univariate_demo.ipynb` としてリネーム/整理する。
2. 依存ライブラリ（`pandas`, `sweetviz` 等）のインポートを整理する。

**Relevant files**
- `KABUCALIB/src/analysis/univariate.py` — **(新設)** 分析ロジックの本体。
- `c:\Users\omoy1\Desktop\KABUCALIB\univariate_analysis copy.ipynb` — 既存ロジックの抽出元であり、最終的に `univariate_demo.ipynb` へ移行。

**Verification**
1. `analyze_feature` の戻り値（DataFrame）が、移行前後のNotebook実行結果で完全に一致することを確認する。
2. `outputs/univariate/` に期待通りのCSVファイルが生成されることを確認する。
3. `univariate_demo.ipynb` が最小限のコード（インポート $\rightarrow$ 関数呼び出し $\rightarrow$ 表示）で動作することを確認する。

**Decisions**
- **スコープ**: 今回は単変量分析と2変数間のクロス分析までをライブラリ化し、`sweetviz` などの外部ライブラリによる可視化はNotebook側に残す。
- **デフォルト値**: `bins` のデフォルト値は、Notebookの実装に合わせて単変量=5, 2変数=3 とする。 

---

## Plan: Univariate分析のライブラリ化とNotebook整理 (Revised)

**「やりすぎない」設計思想**
`analyze_pair` を後回しにする判断は賢明です。単変量分析は出力が定型的なため、まずはここを確実に「救出」し、成功体験を作ることで、その後の拡張（2変数分析や結果クラスの拡充）を自然な流れで進めることができます。

また、`UnivariateResult` データクラスの導入により、将来的な拡張性（チャートパスやメタデータの保持）を確保しつつ、現状のシンプルさを維持するアプローチを採用します。

Notebook内の分析ロジックを `KABUCALIB/src/analysis/univariate.py` に切り出し、再利用可能な形式にします。分析ロジックは src へ、可視化と試行錯誤は `Notebook` へという境界線を明確にし、段階的に移行します。

**Steps**

### Phase 1: 単一特徴量分析の救出 (PR1)
1. `KABUCALIB/src/analysis/` ディレクトリを作成する。
2. `KABUCALIB/src/analysis/univariate.py` を作成し、以下を実装する。
    - `@dataclass class UnivariateResult`: `feature_name` と `summary_df` を保持する。
    - `analyze_feature(df, feature, ...)`: `UnivariateResult` を返す最小限の分析ロジック。
3. Notebook `univariate_analysis copy.ipynb` でこの関数を呼び出し、既存のインライン実装と結果が一致することを検証する。

### Phase 2: 一括分析の実装 (PR2)
1. `univariate.py` に `analyze_features(df, features, ...)` を追加し、複数特徴量の分析結果を `List[UnivariateResult]` として集約する機能を実装する。

### Phase 3: 出力自動化の実装 (PR3)
1. `univariate.py` に `export_univariate_results(results, path)` を実装し、`KABUCALIB/outputs/univariate/` へのCSV保存を可能にする。

### Phase 4: Notebookの整理 (PR4)
1. `univariate_analysis copy.ipynb` から冗長な計算コードを削除し、可視化専用の `univariate_demo.ipynb` へ移行・整理する。

### Phase 5: 2変数分析への拡張 (PR5 - 必要に応じて)
1. `analyze_pair(df, feature1, feature2, ...)` を実装し、クロス集計ロジックをライブラリ化する。

**Relevant files**
- `KABUCALIB/src/analysis/univariate.py` — **(新設)** 分析ロジックの本体。1ファイルで完結させる。
- `c:\Users\omoy1\Desktop\KABUCALIB\univariate_analysis copy.ipynb` $\rightarrow$ `univariate_demo.ipynb`

**Verification**
1. `analyze_feature` の戻り値が移行前後で一致すること。
2. `analyze_features` で指定した全特徴量の結果が正しく集約されること。
3. `outputs/univariate/` に期待通りのCSVが生成されること。

**Decisions**
- **設計**: 複雑なディレクトリ構造（models/services等）は避け、`univariate.py` 1ファイルに集約する。
- **データ構造**: 単なるDataFrameではなく `UnivariateResult` クラスを導入し、将来の拡張に備える。
- **優先順位**: `analyze_pair` は出力形式が多様であるため、単変量分析の基盤が固まった後に着手する。

---

この「小さく切り出す」プランで進めます。まずは **Phase 1 (PR1): `analyze_feature` の救出** から着手しましょう。