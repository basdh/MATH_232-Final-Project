# Directory File Reference (Comprehensive)

This document explains **every top-level file** currently present in this directory, with emphasis on:

- motivation (why the file exists),
- where it sits in the pipeline,
- data structures and features (for CSVs),
- functionality and outputs (for code files),
- what the run progression typically looks like end-to-end.

---

## 1) End-to-end motivation and progression

The project supports two complementary modeling workflows on Polymarket data:

1. **Active-classified workflow (main current workflow)**
   - Pull active markets by category/tag.
   - Build both a raw time-grid dataset and per-market engineered features.
   - Analyze with:
     - **PCA** (linear geometry of feature vectors), and
     - **Autoencoders/CVAE** (nonlinear latent spaces from time series).

2. **Sports closed-market workflow (earlier/parallel workflow)**
   - Pull recently closed head-to-head sports markets.
   - Build sports-specific feature vectors and metadata.

Typical progression to run the current main workflow:

1. Configure + pull active data:
   - `python fetch_polymarket_active_classified.py` (interactive), or
   - `python fetch_polymarket_active_classified.py --no-prompt` (batch mode).
2. Generate PCA visuals from features:
   - `python pca_plot_active_classified_features.py`
3. Generate nonlinear latent visuals from raw sequences:
   - `python autoencoder_timeseries_active_classified.py --mode all`
4. Inspect produced HTML files in browser.

---

## 2) Code files (.py)

### `fetch_polymarket_active_classified.py`
**Purpose:** Main extraction + feature-engineering pipeline for active markets grouped by classifier labels (Sports, Politics, Crypto, etc.).

**Core functionality:**
- Reads defaults from `polymarket_active_classified_config.json`.
- Discovers active Gamma markets by tag with pagination and optional prefilters.
- Pulls CLOB histories (batch first, then per-token fallback).
- Uses native interval preference without mixing granularities in a single synthetic history (`1m` -> `max` -> `1h` fallback logic).
- Resamples to a configurable UTC grid (`interval_hours`) using bucket-last semantics.
- Preserves missing buckets as NaN (no artificial interpolation across empty buckets).
- Optionally captures order-book snapshot features.
- Computes per-market scalar feature vector (`feat_*`) for PCA.
- Writes three aligned outputs atomically:
  - raw grid rows,
  - one-row-per-market metadata,
  - one-row-per-market PCA features (Z-scored across markets when possible).

**Outputs produced by this script in this directory:**
- `polymarket_active_classified_raw_30m.csv`
- `polymarket_active_classified_metadata.csv`
- `polymarket_active_classified_features.csv`

---

### `pca_plot_active_classified_features.py`
**Purpose:** PCA and visualization engine for feature-based market comparison.

**Core functionality:**
- Loads features/metadata CSV input.
- Supports category inclusion filtering:
  - explicit `--include-classifiers`, or
  - interactive per-category Y/N prompts (default Y in TTY mode).
- Supports market-id exclusions from CLI or file.
- Can export IDs from dropped categories for iterative reruns.
- Selects numeric feature columns automatically (excluding id/label columns).
- Fits PCA (configurable max component count).
- Writes primary interactive scatter and optional sidecar analyses:
  - scree plot,
  - loadings structure plot,
  - pairwise distance heatmap.
- Distance heatmap now uses category-aware labels (`Category: market_id`).

**Outputs produced by this script in this directory:**
- `polymarket_active_classified_pca_scatter.html`
- `polymarket_active_classified_pca_scatter_scree.html`
- `polymarket_active_classified_pca_scatter_loadings.html`
- `polymarket_active_classified_pca_scatter_distances.html`
- `polymarket_active_classified_pca_scatter.png`
- plus category-specific scatter variants (e.g. `...__Politics.html`).

---

### `autoencoder_timeseries_active_classified.py`
**Purpose:** Learns nonlinear latent representations from per-market price trajectories.

**Core functionality:**
- Reads raw active time-grid CSV.
- Pivots rows into market-by-step sequences.
- Builds rich inputs by concatenating:
  - price levels, and
  - first differences.
- Standardizes features for stable training.
- Supports multiple model modes:
  - unsupervised deep autoencoder,
  - semi-supervised autoencoder (reconstruction + class CE loss),
  - conditional VAE (CVAE) with KL regularization.
- Exports embeddings CSVs and Plotly latent-space HTML files.

**Outputs produced by this script in this directory:**
- `polymarket_active_classified_autoencoder_embeddings.csv`
- `polymarket_active_classified_autoencoder_embeddings_semi.csv`
- `polymarket_active_classified_cvae_embeddings.csv`
- `polymarket_active_classified_autoencoder_latent.html`
- `polymarket_active_classified_autoencoder_latent_semi.html`
- `polymarket_active_classified_cvae_latent.html`

---

### `fetch_polymarket_sports_pca.py`
**Purpose:** Parallel/legacy sports-focused extraction pipeline for recently closed head-to-head markets.

**Core functionality:**
- Discovers closed sports markets from Gamma (head-to-head filters).
- Pulls CLOB histories around inferred end windows.
- Produces sports-specific features and metadata.
- Supports time-window and skew-threshold controls.

**Outputs produced by this script in this directory:**
- `polymarket_sports_pca_features.csv`
- `polymarket_sports_pca_metadata.csv`
- `polymarket_sports_raw_timeseries.csv`

---

## 3) CSV files (structure + features)

### `polymarket_active_classified_raw_30m.csv` (114,876 rows, 32 columns)
**Role:** Long-format, per-market per-time-bucket dataset (rawest active-classified artifact used for sequence modeling).

**Schema:**
- Market identity/context:
  - `market_id`, `classifier_label`, `tag_id`, `tag_slug`, `question`
- Time axis:
  - `timestamp_iso`, `unix_ts`, `normalized_step`, `lookback_step`, `hours_since_market_start`, `gamma_market_start_iso`
- Price fields:
  - `price_yes`, `price_yes_norm`
- Grid config lineage:
  - `grid_interval`, `interval_hours`, `lookback_days`
- Intra-bucket diagnostics:
  - `raw_prints_in_bucket`, `intrabucket_price_range`, `intrabucket_price_stdev`
- Gamma liquidity/volume snapshot fields:
  - `gamma_volume_num`, `gamma_liquidity_num`, `gamma_volume`, `gamma_liquidity_clob`
- Optional order-book snapshot fields:
  - `ob_best_bid`, `ob_best_ask`, `ob_mid`, `ob_spread`, `ob_imbalance_topn`, `ob_bid_depth_topn`, `ob_ask_depth_topn`, `ob_timestamp_ms`
- Misc:
  - `note`

**Why it matters:**
- Preserves temporal trajectory information needed by autoencoders.
- Supports per-bucket activity/quality checks.

---

### `polymarket_active_classified_metadata.csv` (240 rows, 48 columns)
**Role:** One-row-per-market enriched metadata table combining identity, diagnostics, snapshots, and engineered features.

**Schema families:**
- Identity and labels:
  - `market_id`, `classifier_label`, `tag_id`, `tag_slug`, `question`
- Market timing and platform metrics:
  - `startDate`, `endDate`, `volumeNum`, `liquidityNum`
- Extraction lineage/diagnostics:
  - `interval_hours`, `lookback_days`, `lookback_hours_used`, `clob_points_raw`, `grid_rows`, `grid_freq`, `ob_note`, `price_norm_anchor`
- Gamma snapshots:
  - `gamma_volume_num`, `gamma_liquidity_num`, `gamma_volume`, `gamma_liquidity_clob`
- Order-book snapshots:
  - `ob_best_bid`, `ob_best_ask`, `ob_mid`, `ob_spread`, `ob_bid_depth_topn`, `ob_ask_depth_topn`, `ob_imbalance_topn`, `ob_timestamp_ms`
- Engineered features:
  - `feat_jump_zcount_ge_2`, `feat_jump_zrate_ge_2`,
  - `feat_jump_zcount_ge_3`, `feat_jump_zrate_ge_3`,
  - `feat_jump_count_ge_p95`, `feat_jump_rate_ge_p95`,
  - `feat_jump_count_ge_p99`, `feat_jump_rate_ge_p99`,
  - `feat_jump_max_abs`, `feat_jump_top3_sum_abs`, `feat_jump_top3_frac_total_move`,
  - `feat_return_volatility`, `feat_time_to_close_hours`, `feat_raw_prints_per_hour`,
  - `feat_ob_spread`, `feat_ob_spread_over_mid`, `feat_ob_imbalance_topn`, `feat_ob_mid`, `feat_ob_log_bid_ask_depth_ratio`

**Why it matters:**
- Canonical reference table for QA, interpretability, and joining with outputs.

---

### `polymarket_active_classified_features.csv` (240 rows, 23 columns)
**Role:** PCA-ready compact feature matrix (one row per market).

**Columns:**
- IDs/labels:
  - `market_id`, `classifier_label`, `tag_id`, `tag_slug`
- Core engineered features (`feat_*`, same family as metadata subset).

**Why it matters:**
- Main PCA input with cleaner geometry than the full metadata table.

---

### `polymarket_active_classified_autoencoder_embeddings.csv` (193 rows, 6 columns)
**Role:** Unsupervised autoencoder latent coordinates.

**Columns:**
- `market_id`, `classifier_label`, `tag_slug`, `question`, `z1`, `z2`

**Why it matters:**
- Nonlinear 2D embedding of market trajectory structure without label supervision.

---

### `polymarket_active_classified_autoencoder_embeddings_semi.csv` (193 rows, 8 columns)
**Role:** Semi-supervised autoencoder latent coordinates.

**Columns:**
- Unsupervised columns + `label_class_index`, `label_class`

**Why it matters:**
- Embedding biased to preserve class separability while reconstructing sequences.

---

### `polymarket_active_classified_cvae_embeddings.csv` (193 rows, 9 columns)
**Role:** Conditional VAE latent outputs (using latent mean `mu`).

**Columns:**
- Semi-supervised columns + `latent_is_cvae_mu`

**Why it matters:**
- Generative latent geometry conditioned on classifier labels.

---

### `polymarket_sports_pca_input.csv` (204 rows, 6 columns)
**Role:** Sports-oriented long-format input table.

**Columns:**
- `timestamp`, `sport`, `game_title`, `outcome`, `price`, `market_id`

**Why it matters:**
- Intermediate sports dataset for historical/feature derivation.

---

### `polymarket_sports_pca_features.csv` (4 rows, 9 columns)
**Role:** Sports one-row-per-market engineered features.

**Columns:**
- `market_id`, `sport`, `resolved_yes`,
- `momentum_1h`, `momentum_6h`, `momentum_24h`,
- `volatility_24h`, `efficiency_ratio`, `ripeness_hours`

**Why it matters:**
- Compact sports feature vectors for PCA/analysis.

---

### `polymarket_sports_pca_metadata.csv` (4 rows, 9 columns)
**Role:** Sports market metadata summary.

**Columns:**
- `id`, `question`, `category`, `volumeNum`, `liquidityNum`, `startDate`, `endDate`, `clobTokenIds`, `sport_bucket`

**Why it matters:**
- Context table paired with sports features.

---

### `polymarket_sports_raw_timeseries.csv` (15 rows, 27 columns)
**Role:** Sports raw segment timeseries around inferred windows.

**Columns (families):**
- identity and context:
  - `market_id`, `sport`, `question`, `primary_outcome`
- timing windows:
  - `gamma_listing_start_iso`, `gamma_listing_end_iso`, `inferred_game_start_iso`, `inferred_game_end_iso`, `skew_segment_status`
- source/config:
  - `clob_sources`, `minute_grid_ffill`
- liquidity/volume:
  - `volume_num`, `liquidity_num`, `liquidity`, `liquidity_clob`, `liquidity_amm`, `volume`, `volume_clob`, `volume_1wk_clob`
- orderbook/time/price:
  - `best_bid`, `best_ask`, `timestamp_iso`, `unix_ts`, `price_primary_token`
- window-relative coordinates:
  - `seconds_since_window_start`, `seconds_until_window_end`
- misc:
  - `note`

**Why it matters:**
- Traceable high-fidelity segment view for sports market behavior.

---

## 4) Visualization and report artifacts

### PCA HTML/PNG outputs
- `polymarket_active_classified_pca_scatter.html`
- `polymarket_active_classified_pca_scatter_scree.html`
- `polymarket_active_classified_pca_scatter_loadings.html`
- `polymarket_active_classified_pca_scatter_distances.html`
- `polymarket_active_classified_pca_scatter.png`
- `polymarket_active_classified_pca_scatter__Politics.html`
- `polymarket_active_classified_pca_scatter__Politics__Sports.html`

**What they represent:**
- interactive PCA projection,
- explained-variance (scree),
- loadings/PC structure,
- pairwise distance heatmap,
- static snapshot image,
- category-filtered specialized views.

---

### Autoencoder/CVAE HTML outputs
- `polymarket_active_classified_autoencoder_latent.html`
- `polymarket_active_classified_autoencoder_latent_semi.html`
- `polymarket_active_classified_cvae_latent.html`

**What they represent:**
- latent scatter projections for each model family (unsupervised, semi-supervised, CVAE).

---

## 5) Configuration and dependency files

### `polymarket_active_classified_config.json`
**Role:** Default runtime settings for active-classified extraction (sample counts, lookback, interval, output paths, thresholds).

**Why it matters:**
- Reproducibility and easier batch execution (`--no-prompt`).

---

### `requirements.txt`
**Role:** Python dependency locklist/light spec for running pipelines and visualizations.

**Why it matters:**
- Portable environment recreation via `pip install -r requirements.txt`.

---

## 6) Documentation and utility text files

### `README.md`
**Role:** Primary project guide and workflow explanation.

### `excluded_market_ids.txt`
**Role:** User-maintained or exported exclusion list for PCA reruns.

### `excluded_market_ids_other_categories.txt`
**Role:** Auto-exported list of market IDs outside selected PCA categories (supports iterative filtering workflows).

### `pca_excluded_market_ids_not_in_selected_categories.txt`
**Role:** Additional exclusion artifact for PCA category-selection workflows.

---

## 7) System artifact

### `.DS_Store`
macOS Finder metadata file. Not part of the analytical pipeline.

---

## 8) Practical run playbook (recommended)

1. **Extract active data**
   - `python fetch_polymarket_active_classified.py`
2. **Generate PCA and diagnostics**
   - `python pca_plot_active_classified_features.py`
3. **(Optional) retrain latent sequence models**
   - `python autoencoder_timeseries_active_classified.py --mode all`
4. **Open HTML outputs**
   - PCA + scree + loadings + distance
   - AE/semi/CVAE latent plots
5. **Iterate with exclusions/category subsets**
   - use exported exclusion TXT files and rerun PCA.

This progression gives both:
- an interpretable, feature-engineered view (PCA), and
- a nonlinear sequence-shape view (autoencoder family).
