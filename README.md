# Prediction Market Classification

This repository pulls Polymarket data from **Gamma** (market metadata) and **CLOB** (price history & order book), builds **tabular summaries** and/or **time series**, then supports **PCA** (linear geometry of feature vectors) and **neural autoencoders** (nonlinear compression of price sequences).

**CSV outputs use atomic replace** (`tempfile` + `os.replace`): if the process crashes, you should not get a half-written file; features and metadata row counts stay aligned.

---

## Overview: two pipelines

| Pipeline | Script | Markets | Main outputs | Downstream |
|----------|--------|---------|--------------|------------|
| **Sports (closed, head-to-head)** | `fetch_polymarket_sports_pca.py` | Recently **closed** moneyline-style markets per sport | `polymarket_sports_pca_features.csv`, metadata, raw timeseries | PCA-style features (doc focus: sports script) |
| **Active + classifiers** | `fetch_polymarket_active_classified.py` | **Live** markets tagged by topic (Sports, Politics, Crypto, …) | Raw grid CSV, metadata, **`feat_*`** features CSV | **`pca_plot_active_classified_features.py`**, **`autoencoder_timeseries_active_classified.py` |

The sections below go **depth-first on the active-classified path** (extraction → processing → PCA & autoencoder), then summarize the **sports** script.

---

## Active classified: end-to-end story

You get **two representations** of the same set of markets:

1. **Tabular `feat_*` (one row per market)** — hand-crafted scalars (jumps, volatility, spread, time-to-close, …). Good for **fast, interpretable** linear structure → **PCA**.
2. **Long raw CSV (many rows per market)** — `price_yes` on a UTC time grid. Good for **sequence shape** → **autoencoder**.

Both are produced by **`fetch_polymarket_active_classified.py`**.

### Configuration

- Defaults live in **`polymarket_active_classified_config.json`** (project directory or next to the script). Typical keys: `markets_per_category`, `interval_hours`, `lookback_days`, `max_scan_offset`, `raw_csv`, `metadata_csv`, `features_csv`, plus volume/liquidity filters and order-book options.
- **Interactive run:** `python fetch_polymarket_active_classified.py` — the script can prompt for each setting; **Enter** keeps the bracketed default.
- **Non-interactive / automation:** `--no-prompt` uses the JSON as-is. **`--config PATH`** selects another JSON file. Piped or non-TTY stdin skips prompts.

### 1) Discovery (Gamma)

For each **built-in classifier** (label + `tag_id` + slug), the script pages **`/markets`** with:

- `closed=false`, `active=true` (active markets),
- the classifier’s **tag**,
- optional **min volume / min liquidity** filters,
- up to **`markets_per_category`** markets per tag, stopping when pagination hits **`max_scan_offset`**.

Classifiers include: Sports, Politics, Crypto, Pop culture, Tech, Finance, Business, Geopolitics (see `DEFAULT_CLASSIFIERS` in the script).

### 2) CLOB price history

- The public API is queried with the **yes-outcome token id** (not the human `market` slug).
- A **batch** request loads history for the global window **[now − lookback_days, now]** (UTC), preferring **1m** data.
- If a token’s batch history is **too sparse**, a per-token fallback **`merge_finest_clob`** runs: try **`1m` → `max` → `1h`** and use the **first** interval that returns a usable series, **without** mixing different resolutions into one synthetic track (avoids blended granularities).
- The pipeline logs progress (windows, token counts, batch chunk completion) so long runs are easier to follow.

### 3) Grid resampling (processing)

- CLOB has **no** arbitrary sub-hour “bucket” for every custom step size; the code **resamples** to your chosen **`interval_hours`** (e.g. 0.5 h) using pandas.
- **`resample_to_price_grid`**:
  - builds a regular UTC index from the lookback window;
  - uses **last trade per bucket** when reindexing;
  - **does not forward-fill** across empty buckets — gaps stay **NaN** so the grid does not invent prices through quiet periods (only what the API returned is represented at that fidelity).
- **`add_normalized_price_column`** adds **`price_yes_norm`**: price changes relative to an **anchor** (first valid price), used for feature math that works in “change” space.
- **`augment_grid_with_bucket_hist`** enriches each grid row with **counts of raw CLOB prints** in that bucket and simple intra-bucket dispersion (so “activity” is visible per cell).

### 4) Order book (optional)

- If enabled, **`/book`** is called once per market. You get a **snapshot** (best bid/ask, spread, mid, top-N depth, imbalance) — **not** a full historical L2 tape. The same snapshot values can be attached to each grid row in the raw CSV for convenience.

### 5) Scalar features (`feat_*`)

**`compute_market_features`** collapses each market’s grid + fine history + (optional) OB into **one vector** of `feat_*` keys, including:

- **Dynamics:** jumps and rates (including z-score-style thresholds, high-percentile jump stats, volatility of returns, max jumps, share of move in top spikes, …).
- **Timing:** **`feat_time_to_close_hours`** from Gamma `endDate`.
- **Activity:** e.g. **`feat_raw_prints_per_hour`** tied to bucket-level print counts.
- **Microstructure (when OB enabled):** spread, spread/mid, imbalance, mid, log depth ratio, etc.

Exact column names are centralized as **`PCA_FEATURE_KEYS`** in `fetch_polymarket_active_classified.py`.

### 6) Output files

| File | Contents |
|------|----------|
| **Raw CSV** (`polymarket_active_classified_raw_*.csv` by default) | Long format: one row per **market × time bucket** with `price_yes`, step indices (`lookback_step` / `normalized_step`), `classifier_label`, `tag_slug`, grid metadata, optional OB/Gamma fields, bucket print stats. **Feeds the autoencoder.** |
| **Metadata CSV** | One row per **market**: question, dates, volume/liquidity, diagnostics (`clob_points_raw`, `grid_rows`, …), snapshot OB, and all **`feat_*`** scalars. |
| **Features CSV** | One row per **market**: `market_id`, `classifier_label`, `tag_id`, `tag_slug`, plus **`feat_*`** only — the slice intended for **PCA**. |

### 7) Z-scoring (features CSV)

- **`zscore_features_for_pca`** applies **`sklearn.preprocessing.StandardScaler`** **across markets** to the PCA feature columns when there are **≥ 2** rows.
- With a single row, scaling is skipped (variance undefined).

---

## PCA on tabular features (`pca_plot_active_classified_features.py`)

**Purpose:** Linear dimensionality reduction on **one vector per market** (the `feat_*` columns). You see which directions explain most variance and how categories separate in PC space.

### Input

- Default **`--input`** is **`polymarket_active_classified_features.csv`**.
- You may point at **`polymarket_active_classified_metadata.csv`** instead; then **every numeric column** except identifiers may enter PCA (see below).

### Feature selection

- Uses **all numeric columns** except: `market_id`, `tag_id`, `classifier_label`, `tag_slug`.
- If nothing qualifies, falls back to **columns after `tag_slug`** (legacy layout).

### Category filtering

- **`--include-classifiers "Sports,Politics"`** — case-insensitive include list; no prompts.
- **Interactive default:** if stdin is a **TTY** and you did not pass `--include-classifiers`, the script asks **Y/n for each category** (default **Y**).
- **`--no-category-prompt`** — include all categories without asking (scripts / CI).
- Non-TTY stdin falls back to **all categories** with a stderr note (or use explicit flags).
- Dropped categories can write **`excluded_market_ids_other_categories.txt`** (market ids for “other” categories) unless **`--no-write-excluded`**. Use **`--list-classifiers`** to print labels from the CSV.
- **`--exclude-market-ids`** / **`--exclude-market-ids-file`** remove specific markets after category filtering.

### Fitting

- Fits PCA with up to **`--max-pca-components`** (default **20**, capped by sample and feature counts).
- Main scatter uses **PC1 vs PC2**. Prints **feature–PC correlations** in the terminal.
- Optional **`--metadata`** supplies **`question`** for richer Plotly hover (auto-detects `polymarket_active_classified_metadata.csv` if omitted).

### Visualizations (Plotly HTML)

Sidecar files share the **same path stem** as **`--output`** (e.g. `out.html` → `out_scree.html`). Disable all extras with **`--no-extra-plots`**.

| File | What it shows |
|------|----------------|
| **`--output`** (default `polymarket_active_classified_pca_scatter.html`) | **2D scatter** (PC1 vs PC2), color = `classifier_label`. Toolbar zoom/pan/lasso; optional panel to export selected **`market_id`** for a future exclusion run. |
| **`*_scree.html`** | **Explained variance** per PC (bars) + **cumulative** variance (line). Answers “how many components matter?” |
| **`*_loadings.html`** | **Heatmap:** features × PCs (sklearn **components_**). Second panel: either **PC scores** along markets **sorted by `feat_time_to_close_hours`** (if present), or **loadings vs PC index** for top features. |
| **`*_distances.html`** | **Pairwise Euclidean distances** between markets. Default **`--distance-space latent`** uses the **first K PC scores** (`--distance-pc-dims`, default 10). **`raw`** uses the same numeric feature matrix as PCA. Axis labels **`Category: market_id`**. Large **N** subsampled (`--heatmap-max-markets`, default 80). |

**PNG output:** pass **`--output something.png`** for a **static matplotlib** scatter only (extra Plotly pages are still written unless `--no-extra-plots`).

---

## Autoencoder on time series (`autoencoder_timeseries_active_classified.py`)

**Purpose:** Nonlinear **compression of entire price paths** (plus first differences), optionally **using category labels** in the loss. Complementary to PCA: PCA summarizes scalars; the autoencoder learns structure in **sequences**.

### Input

- Default **`--raw-input`** is **`polymarket_active_classified_raw_30m.csv`** (or whatever raw file your fetch step produced).

### Processing

1. **`build_timeseries_matrix`** — pivots **`price_yes`** to **market × time step** (`lookback_step` or `normalized_step`). Drops markets with too few points (`--min-points-per-market`). Applies light along-sequence fill so the pivot is usable.
2. **`build_rich_features`** — concatenates **[price levels | first differences]** into one wide vector per market.
3. **`StandardScaler`** (+ clipping) on those vectors.
4. **`LabelEncoder`** on **`classifier_label`** for supervised variants.

### Models (`--mode`)

| Mode | Idea |
|------|------|
| **`unsupervised`** | Deep **tanh** encoder–decoder; loss = reconstruction **MSE**. |
| **`semi-supervised`** | Same network + linear **classifier** on latent: **MSE + α × cross-entropy** (`--semi-weight`). |
| **`cvae`** | **Conditional VAE**: encoder/decoder see **label one-hot**; loss includes **β × KL** (`--beta-kl`). Plots use latent **μ**. |
| **`both` / `all`** | Runs combinations as implemented (see `--help`). |

### Outputs

- **Plotly HTML** — scatter of **latent z₁ vs z₂** (default **`--latent-dim 2`**), colored by category, hover **`market_id`**, **`tag_slug`**, **`question`**.
- **CSV embeddings** — latent coordinates + metadata (`--embeddings-csv`, plus `_semi` / `_cvae` variants).

---

## PCA vs autoencoder (how to choose)

| | **PCA (`feat_*`)** | **Autoencoder (raw series)** |
|--|----------------------|-------------------------------|
| **Input** | One row per market, fixed **`feat_*`** vector | Many rows per market: **discretized `price_yes`** |
| **Linear / nonlinear** | **Linear** subspace | **Nonlinear** (tanh nets / VAE) |
| **Labels** | Only for **coloring / filtering** plots | Optional **semi-supervised** or **CVAE** losses |
| **Use when** | You care about **interpretable summaries** and fast **2D maps** of those summaries | You care about **shape of the curve** and **learned** latent geometry |

---

## Sports pipeline (`fetch_polymarket_sports_pca.py`) — summary

**Different use case:** **closed**, **head-to-head** sports markets (team vs team), not the active-classified topic tags.

```bash
pip install -r requirements.txt
python fetch_polymarket_sports_pca.py           # default 20 markets per sport
python fetch_polymarket_sports_pca.py 15        # positional: 15 per sport
python fetch_polymarket_sports_pca.py --markets-per-sport 10
```

### CLI (sports)

| Argument | Meaning |
|----------|--------|
| `N` (optional positional) | Markets per sport (same as `--markets-per-sport`). Default **20**. |
| `--markets-per-sport N` | Same. |
| `--per-sport N` | Deprecated alias. |
| `--max-scan-offset` | Max Gamma offset scanned **per sport** (default **1200**). |
| `--features-out` / `--metadata-out` | Output CSV paths. |
| `--raw-timeseries-out` | Long-format prices (default `polymarket_sports_raw_timeseries.csv`). |
| `--market-end-within-last-days` | `endDate` within last *N* days (UTC). Default **30**; **0** disables. |
| `--market-end-after` / `--market-end-before` | Optional ISO bounds on `endDate`. |
| `--skew-lookback-hours` | Hourly CLOB before `endDate` for **resolution skew** heuristic (default **336**). |
| `--skew-high` / `--skew-low` | Implied prob thresholds (default **0.95** / **0.05**). |
| `--raw-segment-hours` | Length of **1m** pull ending at inferred game end (default **2** h). |
| `--raw-minute-grid` | Optional **1-minute** forward-fill inside the segment. |
| `--raw-fetch-lookback-hours` | CLOB lookback for **PCA feature** merge (default **240** h). |

### Sports: discovery

1. **`/markets`** with league **`tag_id`**, `closed=true`, recent **`endDate` first**.
2. If needed, **`/events`** with **`series_id`** for embedded moneylines (soccer uses several series ids).

**Head-to-head filter:** title must look like **“Team A vs. Team B”**; excludes **“Will …”**, spreads, obvious O/U. **`endDate` &lt; now (UTC).**

### Sports: CLOB

- **`market=<asset_id>`** for the **first-listed** team’s token.
- **Features:** bounded lookback before `endDate` with **`1h` + `1m`** merge; optional unbounded call if &lt; 2 points.
- **Raw timeseries:** hourly skew detection → inferred end → **`1m`** segment of length **`--raw-segment-hours`**.

### Sports: outputs

- **`polymarket_sports_pca_features.csv`** — one row per market; Z-scored when **≥ 2** rows.
- **`polymarket_sports_pca_metadata.csv`** — same count + skew / inferred window fields when detection succeeds.
- **`polymarket_sports_raw_timeseries.csv`** — long-format **1m** segment + Gamma columns.

**Features (48h before `endDate`):** logit implied prob (clipped), **`resolved_yes`**, `momentum_*`, `volatility_24h`, `efficiency_ratio`, `ripeness_hours` (see script docstring). With **≥ 2** rows, **`StandardScaler`** on the six numeric PCA columns; one row → no scaling.

---

## Quick reference: active-classified commands

```bash
# Fetch (config + optional prompts)
python fetch_polymarket_active_classified.py
python fetch_polymarket_active_classified.py --no-prompt

# PCA + scree + loadings + distance heatmaps (default outputs next to --output stem)
python pca_plot_active_classified_features.py --no-category-prompt

# Autoencoder (example)
python autoencoder_timeseries_active_classified.py --mode both --epochs 1200
```

Install dependencies: **`pip install -r requirements.txt`**.
