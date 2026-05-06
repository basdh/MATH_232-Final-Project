#!/usr/bin/env python3
"""
PCA on polymarket_active_classified_features.csv: each row is a market (feature vector);
columns after ``tag_slug`` are numeric features (typically Z-scored).

Default output is an **interactive** Plotly HTML file: hover shows market info; use the
toolbar (box zoom, pan, lasso) or scroll wheel (enabled) to focus on a region. Double‑click
resets the axes.

When you run this script in a terminal **without** ``--include-classifiers``, it prompts
**Y/n for each category** (default Y). Use ``--no-category-prompt`` to include all categories
without prompting.

**Outliers:** use toolbar **Box Select** or **Lasso Select**, select points you treat as
outliers; their ``market_id`` values appear in the panel below. Download
``excluded_market_ids.txt`` and rerun with ``--exclude-market-ids-file`` to refit PCA without
those rows (see ``--help``).

Pass ``--output something.png`` for a static matplotlib figure instead.

Also writes three Plotly HTML pages next to the main ``--output`` stem:

- ``*_scree.html`` — explained variance per PC plus cumulative curve.
- ``*_loadings.html`` — feature×PC loadings heatmap; second panel is PC scores vs markets
  sorted by ``feat_time_to_close_hours`` when that column exists (otherwise loadings vs PC index).
- ``*_distances.html`` — pairwise Euclidean distances between markets (latent first-K PCs or raw features).

Use ``--no-extra-plots`` to skip these. Distance matrices subsample with ``--heatmap-max-markets``.
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import sys
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def feature_columns_after_tag_slug(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    if "tag_slug" not in cols:
        raise ValueError("Expected column 'tag_slug' in CSV.")
    i = cols.index("tag_slug") + 1
    feats = cols[i:]
    if not feats:
        raise ValueError("No feature columns after 'tag_slug'.")
    return feats


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Use all numeric columns except explicit identifiers/labels.
    This keeps liquidity/volume and other numeric metadata in PCA.
    """
    exclude = {
        "market_id",
        "tag_id",
        "classifier_label",
        "tag_slug",
    }
    numeric_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        # Last-resort fallback for files where numerics were parsed as objects.
        fallback = feature_columns_after_tag_slug(df)
        if not fallback:
            raise ValueError("No usable numeric columns found for PCA.")
        return fallback
    return numeric_cols


def resolve_metadata_csv(features_path: str, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        p = os.path.abspath(explicit)
        return p if os.path.isfile(p) else None
    base = os.path.dirname(os.path.abspath(features_path))
    for cand in (
        os.path.join(base, "polymarket_active_classified_metadata.csv"),
        os.path.join(os.getcwd(), "polymarket_active_classified_metadata.csv"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def merge_question(df: pd.DataFrame, meta_path: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    out["_question"] = ""
    if not meta_path:
        return out
    try:
        m = pd.read_csv(meta_path, usecols=lambda c: c in ("market_id", "question"))
    except ValueError:
        m = pd.read_csv(meta_path)
        if "question" not in m.columns:
            return out
        m = m[["market_id", "question"]]
    m = m.drop_duplicates(subset=["market_id"], keep="first")
    qmap = dict(zip(m["market_id"].astype(str), m["question"].astype(str).fillna("")))
    out["_question"] = out["market_id"].astype(str).map(qmap).fillna("")
    return out


def parse_include_classifiers(s: Optional[str]) -> Optional[set]:
    """Return set of allowed classifier_label values (lowercased for matching), or None = all."""
    if not s or not str(s).strip():
        return None
    out: set = set()
    for part in str(s).replace(";", ",").split(","):
        p = part.strip()
        if p:
            out.add(p.lower())
    return out if out else None


def _yes_default_y(s: str) -> bool:
    """Empty input means Yes; only explicit n/no excludes."""
    t = (s or "").strip().lower()
    if not t:
        return True
    return t not in ("n", "no")


def prompt_category_inclusion(sorted_labels: List[str]) -> set:
    """
    Ask Y/N for each classifier label (default Y). Returns lowercase strings to keep.
    """
    included: set = set()
    print("\nWhich categories should be included in the PCA? (default: Y — press Enter)\n")
    for lab in sorted_labels:
        display = lab if lab else "(empty)"
        try:
            ans = input(f"  Include '{display}'? [Y/n]: ")
        except EOFError:
            ans = ""
        if _yes_default_y(ans):
            included.add(str(lab).strip().lower())
    return included


def filter_by_classifiers(
    df: pd.DataFrame, allowed_lower: set
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep rows whose classifier_label matches allowed (case-insensitive).
    Returns (kept_df, dropped_df).
    """
    if "classifier_label" not in df.columns:
        raise SystemExit("Input CSV has no 'classifier_label' column; cannot filter by category.")
    lab = df["classifier_label"].astype(str).str.strip()
    mask = lab.str.lower().isin(allowed_lower)
    return df.loc[mask].reset_index(drop=True), df.loc[~mask].reset_index(drop=True)


def parse_exclude_ids(comma: Optional[str], file_path: Optional[str]) -> Set[str]:
    out: Set[str] = set()
    if comma:
        for part in comma.replace(",", " ").split():
            p = part.strip()
            if p:
                out.add(p)
    if file_path:
        pth = os.path.abspath(file_path)
        if not os.path.isfile(pth):
            raise SystemExit(f"--exclude-market-ids-file not found: {pth}")
        with open(pth, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                for tok in line.replace(",", " ").split():
                    t = tok.strip()
                    if t:
                        out.add(t)
    return out


# Injected after Plotly loads; Box/Lasso selection fills textarea for rerun with --exclude-market-ids-file
OUTLIER_PANEL_POST_SCRIPT = """
(function () {
  function init() {
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) { setTimeout(init, 100); return; }
    var panel = document.createElement('div');
    panel.style.cssText = 'margin:16px 12px 24px;font-family:system-ui,sans-serif;max-width:920px;border-top:1px solid #ccc;padding-top:12px;';
    panel.innerHTML =
      '<h3 style="margin:6px 0 8px;font-size:1.05rem;">Outlier selection (optional)</h3>' +
      '<p style="margin:6px 0;color:#444;line-height:1.45;font-size:13px;">' +
      'Switch the toolbar from <b>Zoom</b> to <b>Box Select</b> or <b>Lasso Select</b>, then drag around points you want to <b>exclude</b> from a future PCA run. ' +
      'Their <code>market_id</code> values appear below (each new selection replaces the list). ' +
      'Download the file and rerun: <code>python pca_plot_active_classified_features.py --exclude-market-ids-file excluded_market_ids.txt</code>' +
      '</p>' +
      '<label for="pca-excluded-ids-textarea" style="display:block;margin-top:8px;font-weight:600;font-size:13px;">Selected market IDs to exclude next run</label>' +
      '<textarea id="pca-excluded-ids-textarea" rows="7" style="width:100%;font-family:ui-monospace,monospace;font-size:12px;margin-top:6px;box-sizing:border-box;"></textarea>' +
      '<div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;">' +
      '<button type="button" id="pca-download-excluded" style="padding:6px 12px;">Download excluded_market_ids.txt</button>' +
      '<button type="button" id="pca-clear-selection" style="padding:6px 12px;">Clear list</button>' +
      '</div>';

    document.body.appendChild(panel);

    gd.on('plotly_selected', function (ev) {
      if (!ev || !ev.points || !ev.points.length) return;
      var ids = ev.points.map(function (pt) { return String(pt.customdata[0]); });
      var seen = {};
      var uniq = [];
      for (var i = 0; i < ids.length; i++) {
        if (!seen[ids[i]]) { seen[ids[i]] = true; uniq.push(ids[i]); }
      }
      var ta = document.getElementById('pca-excluded-ids-textarea');
      if (ta) ta.value = uniq.join('\\n');
    });

    document.getElementById('pca-download-excluded').onclick = function () {
      var ta = document.getElementById('pca-excluded-ids-textarea');
      var text = ta ? ta.value : '';
      var blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'excluded_market_ids.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
    };
    document.getElementById('pca-clear-selection').onclick = function () {
      var ta = document.getElementById('pca-excluded-ids-textarea');
      if (ta) ta.value = '';
    };
  }
  if (document.readyState === 'complete') init();
  else window.addEventListener('load', init);
})();
"""


def write_plotly_html(
    plot_df: pd.DataFrame,
    ev: np.ndarray,
    out_path: str,
    *,
    title_suffix: str = "",
    include_outlier_panel: bool = True,
) -> None:
    try:
        import plotly.express as px
    except ImportError as e:  # pragma: no cover
        print("Install plotly: pip install plotly", file=sys.stderr)
        raise SystemExit(1) from e

    def esc(s: str) -> str:
        return html_module.escape(str(s), quote=True)

    plot_df = plot_df.copy()
    plot_df["_hover_q"] = plot_df["_question"].map(lambda x: esc(x) if x else "—")

    title = "PCA — active classified markets (rows = samples)" + title_suffix

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="classifier_label",
        custom_data=["market_id", "tag_slug", "_hover_q", "classifier_label"],
        title=title,
        labels={
            "PC1": f"PC1 ({ev[0] * 100:.1f}% var)",
            "PC2": f"PC2 ({ev[1] * 100:.1f}% var)",
        },
    )
    fig.update_traces(
        marker=dict(size=9, line=dict(width=0.6, color="rgba(0,0,0,0.45)")),
        hovertemplate=(
            "<b>market_id</b> %{customdata[0]}<br>"
            "<b>classifier</b> %{customdata[3]}<br>"
            "<b>tag_slug</b> %{customdata[1]}<br>"
            "<b>question</b> %{customdata[2]}<extra></extra>"
        ),
    )
    fig.update_layout(
        legend_title_text="classifier_label",
        dragmode="zoom",
        hovermode="closest",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor="rgba(128,128,128,0.5)")
    fig.update_yaxes(showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor="rgba(128,128,128,0.5)")

    kwargs = dict(
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False,
        },
        include_plotlyjs="cdn",
    )
    if include_outlier_panel:
        kwargs["post_script"] = OUTLIER_PANEL_POST_SCRIPT

    fig.write_html(out_path, **kwargs)


def _plotly_write_html(fig, out_path: str) -> None:
    fig.write_html(
        out_path,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False,
        },
        include_plotlyjs="cdn",
    )


def max_n_components(n_samples: int, n_features: int, cap: int) -> int:
    m = min(n_samples, n_features, max(1, cap))
    return max(1, m)


def write_scree_html(
    explained_variance_ratio: np.ndarray,
    out_path: str,
    *,
    title_suffix: str = "",
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover
        print("Install plotly: pip install plotly", file=sys.stderr)
        raise SystemExit(1) from e

    ev = np.asarray(explained_variance_ratio, dtype=float)
    k = len(ev)
    pc_labels = [f"PC{i + 1}" for i in range(k)]
    cum = np.cumsum(ev)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pc_labels,
            y=ev,
            name="Explained variance ratio",
            marker_color="rgba(55, 128, 189, 0.85)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pc_labels,
            y=cum,
            mode="lines+markers",
            name="Cumulative explained variance",
            yaxis="y2",
            line=dict(color="rgba(200, 80, 80, 0.9)", width=2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title="Scree plot — variance per principal component" + title_suffix,
        xaxis=dict(title="Component"),
        yaxis=dict(title="Explained variance ratio", rangemode="tozero", range=[0, min(1.05, float(ev.max()) * 1.15 + 0.02)]),
        yaxis2=dict(
            title="Cumulative explained variance",
            overlaying="y",
            side="right",
            range=[0, 1.05],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    _plotly_write_html(fig, out_path)


def write_loadings_html(
    components: np.ndarray,
    feat_cols: List[str],
    out_path: str,
    *,
    title_suffix: str = "",
    df_rows: Optional[pd.DataFrame] = None,
    Z: Optional[np.ndarray] = None,
    time_order_col: str = "feat_time_to_close_hours",
) -> None:
    """
    Row 1: heatmap of sklearn PCA components (features × PCs).
    Row 2: either PC scores along markets sorted by ``feat_time_to_close_hours`` when present,
    or loadings vs PC index for the top features by L2 norm across PCs.

    Rows are markets (not clock time); sorting by time-to-close gives an interpretable ordering axis.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:  # pragma: no cover
        print("Install plotly: pip install plotly", file=sys.stderr)
        raise SystemExit(1) from e

    comp = np.asarray(components, dtype=float)
    n_pc, n_feat = comp.shape
    pc_labels = [f"PC{i + 1}" for i in range(n_pc)]
    z_heat = comp.T

    has_time_order = (
        df_rows is not None
        and Z is not None
        and time_order_col in df_rows.columns
        and len(df_rows) == len(Z)
    )

    if has_time_order:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.55, 0.45],
            vertical_spacing=0.12,
            subplot_titles=(
                "PC loadings (feature weights on each component)",
                f"PC scores along markets ordered by {time_order_col} (index = rank after sort)",
            ),
        )
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.55, 0.45],
            vertical_spacing=0.12,
            subplot_titles=(
                "PC loadings (feature weights on each component)",
                "Loadings vs component index (top features by L2 norm across PCs)",
            ),
        )

    fig.add_trace(
        go.Heatmap(
            z=z_heat,
            x=pc_labels,
            y=feat_cols,
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title="Loading"),
            hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if has_time_order:
        tcol = pd.to_numeric(df_rows[time_order_col], errors="coerce")
        order = np.argsort(tcol.to_numpy())
        Zs = Z[order, :]
        n_show = min(Zs.shape[1], 8)
        x_idx = np.arange(len(order))
        for i in range(n_show):
            fig.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=Zs[:, i],
                    mode="lines",
                    name=f"PC{i + 1}",
                    line=dict(width=1.5),
                    opacity=0.88,
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(title_text="PC score", row=2, col=1)
        fig.update_xaxes(title_text="Sorted market index", row=2, col=1)
    else:
        norms = np.linalg.norm(comp, axis=0)
        top_idx = np.argsort(norms)[::-1][: min(8, n_feat)]
        for j in top_idx:
            fig.add_trace(
                go.Scatter(
                    x=pc_labels,
                    y=comp[:, j],
                    mode="lines+markers",
                    name=feat_cols[j][:40],
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(title_text="Loading", row=2, col=1)
        fig.update_xaxes(title_text="Principal component", row=2, col=1)

    fig.update_layout(
        title="PCA loadings and component structure" + title_suffix,
        height=880,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )
    _plotly_write_html(fig, out_path)


def write_distance_heatmap_html(
    Z_dist: np.ndarray,
    market_ids: List[str],
    out_path: str,
    *,
    categories: Optional[List[str]] = None,
    title_suffix: str = "",
    space_desc: str = "latent (PC scores)",
) -> None:
    """Pairwise Euclidean distances; axes/hover use ``category: market_id`` when categories are given."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover
        print("Install plotly: pip install plotly", file=sys.stderr)
        raise SystemExit(1) from e

    D = pairwise_distances(Z_dist, metric="euclidean")
    n = D.shape[0]
    if categories is not None and len(categories) == n:
        labels = [f"{str(categories[i])}: {market_ids[i]}" for i in range(n)]
    else:
        labels = [str(m) for m in market_ids]

    fig = go.Figure(
        data=go.Heatmap(
            z=D,
            x=labels,
            y=labels,
            colorscale="Viridis",
            hovertemplate="%{x}<br>vs %{y}<br>distance %{z:.4f}<extra></extra>",
        )
    )
    side = min(950, 420 + n * 6)
    fig.update_layout(
        title=f"Pairwise market distances ({space_desc})" + title_suffix,
        xaxis=dict(title="category : market_id", tickangle=-45),
        yaxis=dict(title="category : market_id", autorange="reversed"),
        height=side,
        width=side,
    )
    if n > 45:
        step = max(1, n // 35)
        tick_idx = list(range(0, n, step))
        fig.update_xaxes(tickvals=tick_idx, ticktext=[labels[i] for i in tick_idx])
        fig.update_yaxes(tickvals=tick_idx, ticktext=[labels[i] for i in tick_idx])
    _plotly_write_html(fig, out_path)


def write_matplotlib_png(plot_df: pd.DataFrame, ev: np.ndarray, out_path: str, dpi: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        raise SystemExit(1) from e

    labels = plot_df["classifier_label"].astype(str)
    uniq = sorted(labels.unique())
    cmap = plt.colormaps["tab10"]
    label_to_color = {lab: cmap((i % 10) / 9.0) for i, lab in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    for lab in uniq:
        mask = labels == lab
        ax.scatter(
            plot_df.loc[mask, "PC1"],
            plot_df.loc[mask, "PC2"],
            c=[label_to_color[lab]],
            label=lab,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
            s=55,
        )
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title("PCA — active classified markets (rows = samples)")
    ax.legend(title="classifier_label", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def print_pc_feature_correlations(X: np.ndarray, Z: np.ndarray, feat_cols: List[str], top_n: int = 8) -> None:
    """
    Print highest absolute Pearson correlations between original features and PC scores.
    """
    if X.shape[0] < 3:
        print("Too few rows to compute stable feature-PC correlations.")
        return
    Xdf = pd.DataFrame(X, columns=feat_cols)
    n_pc = min(2, Z.shape[1])
    print("\nTop feature correlations with principal components:")
    for j in range(n_pc):
        pc_name = f"PC{j + 1}"
        pc_vals = Z[:, j]
        corr_pairs: List[tuple[str, float]] = []
        y = pd.Series(np.asarray(pc_vals).ravel(), dtype=float)
        y_std = float(y.std(ddof=0))
        for c in feat_cols:
            x = pd.to_numeric(Xdf[c], errors="coerce")
            x_std = float(x.std(ddof=0))
            if x_std <= 1e-12 or y_std <= 1e-12:
                corr = float("nan")
            else:
                corr = float(x.corr(y))
            corr_pairs.append((c, corr))
        corr_pairs = [p for p in corr_pairs if not np.isnan(p[1])]
        corr_pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        top = corr_pairs[: max(1, int(top_n))]
        print(f"  {pc_name}:")
        for feat, corr in top:
            print(f"    {feat}: {corr:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PCA scatter of active-classified feature rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outliers / second PCA run:
  1. Open the HTML plot; switch toolbar to Box Select or Lasso Select.
  2. Drag around outlier points; IDs appear in the panel at the bottom.
  3. Click Download excluded_market_ids.txt
  4. Rerun:  %(prog)s --exclude-market-ids-file excluded_market_ids.txt

You can also pass IDs directly:  %(prog)s --exclude-market-ids 558960,559687

Category-restricted PCA (e.g. Sports, Politics, Crypto only):
  %(prog)s --include-classifiers "Sports,Politics,Crypto" --output pca_subset.html
  Writes excluded_market_ids_other_categories.txt (non-selected categories) unless --no-write-excluded.
  List labels in your CSV:  %(prog)s --list-classifiers

Interactive (default when stdin is a terminal): you are prompted Y/n for each category
(default Y). Use --no-category-prompt to include all categories without prompting.
""",
    )
    ap.add_argument(
        "--include-classifiers",
        default=None,
        metavar="NAMES",
        help='Comma-separated classifier_label values to INCLUDE (e.g. "Sports,Politics,Crypto"). '
        "Matching is case-insensitive.",
    )
    ap.add_argument(
        "--write-excluded-market-ids",
        default=None,
        metavar="PATH",
        help="Write market_id values for rows NOT in --include-classifiers (default when filtering: "
        "excluded_market_ids_other_categories.txt). Use --no-write-excluded to skip.",
    )
    ap.add_argument(
        "--no-write-excluded",
        action="store_true",
        help="When using --include-classifiers, do not write excluded market IDs to a file.",
    )
    ap.add_argument(
        "--list-classifiers",
        action="store_true",
        help="Print unique classifier_label values from --input and exit.",
    )
    ap.add_argument(
        "--no-category-prompt",
        action="store_true",
        help="Skip per-category Y/n prompts; include every classifier_label from the CSV. "
        "Use when stdin is not interactive or you want the full dataset without prompts.",
    )
    ap.add_argument(
        "--input",
        default="polymarket_active_classified_features.csv",
        help="Path to features CSV (Z-scored numerics after tag_slug).",
    )
    ap.add_argument(
        "--output",
        default="polymarket_active_classified_pca_scatter.html",
        help="Output path: .html for interactive Plotly (default), .png for static matplotlib.",
    )
    ap.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata CSV with market_id + question for richer hover. "
        "If omitted, looks for polymarket_active_classified_metadata.csv next to --input or cwd.",
    )
    ap.add_argument(
        "--exclude-market-ids",
        default=None,
        metavar="IDS",
        help="Comma- or space-separated market_id values to drop before fitting PCA.",
    )
    ap.add_argument(
        "--exclude-market-ids-file",
        default=None,
        metavar="PATH",
        help="Text file: one market_id per line (optional # comments). Merged with --exclude-market-ids.",
    )
    ap.add_argument(
        "--no-outlier-panel",
        action="store_true",
        help="Omit the HTML panel + JS for exporting selected outliers (smaller page).",
    )
    ap.add_argument("--dpi", type=int, default=120, help="PNG resolution only.")
    ap.add_argument(
        "--max-pca-components",
        type=int,
        default=20,
        metavar="K",
        help="Fit PCA with up to K components (for scree/loadings/distance); capped by min(n, features).",
    )
    ap.add_argument(
        "--no-extra-plots",
        action="store_true",
        help="Do not write sidecar scree, loadings, or distance heatmap HTML next to the main output.",
    )
    ap.add_argument(
        "--heatmap-max-markets",
        type=int,
        default=80,
        metavar="N",
        help="Subsample at most N markets for the pairwise distance heatmap (speed).",
    )
    ap.add_argument(
        "--distance-space",
        choices=("latent", "raw"),
        default="latent",
        help="latent: Euclidean in first K PC scores; raw: Euclidean in feature columns (before PCA).",
    )
    ap.add_argument(
        "--distance-pc-dims",
        type=int,
        default=10,
        metavar="K",
        help="For --distance-space latent, use the first K PC dimensions in the distance matrix.",
    )
    args = ap.parse_args()

    path = os.path.abspath(args.input)
    if not os.path.isfile(path):
        raise SystemExit(f"Input not found: {path}")

    df = pd.read_csv(path)
    if args.list_classifiers:
        if "classifier_label" not in df.columns:
            raise SystemExit("No classifier_label column in CSV.")
        uniq = sorted(df["classifier_label"].astype(str).unique())
        print("classifier_label values in", path)
        for u in uniq:
            print(f"  {u}")
        raise SystemExit(0)

    n_input = len(df)
    cli_inc = args.include_classifiers
    if cli_inc is not None and str(cli_inc).strip():
        allowed = parse_include_classifiers(cli_inc)
    elif cli_inc is not None:
        # Explicit empty '' → include all categories (no filter)
        allowed = None
    elif args.no_category_prompt or not sys.stdin.isatty():
        allowed = None
        if not args.no_category_prompt and not sys.stdin.isatty():
            print(
                "stdin is not a terminal; including all categories. "
                "Use --include-classifiers or --no-category-prompt explicitly.",
                file=sys.stderr,
            )
    else:
        if "classifier_label" not in df.columns:
            raise SystemExit("No classifier_label column; cannot prompt for categories.")
        labels_order = sorted(df["classifier_label"].astype(str).unique())
        picked = prompt_category_inclusion(labels_order)
        allowed = picked if picked else None
        if allowed is None:
            raise SystemExit(
                "No categories selected. Answer Y to at least one category, or use --no-category-prompt / "
                "--include-classifiers."
            )

    if allowed is not None:
        kept, dropped = filter_by_classifiers(df, allowed)
        kept_labels = sorted({str(x) for x in kept["classifier_label"].astype(str).unique()})
        if not kept_labels:
            raise SystemExit(
                "No rows matched --include-classifiers. Check spelling; run with --list-classifiers to see labels."
            )
        print(
            f"Including {len(kept)} row(s) in {kept_labels!s} "
            f"(dropped {len(dropped)} row(s) from other categories)."
        )
        if len(dropped) > 0 and not args.no_write_excluded:
            out_excl = args.write_excluded_market_ids or "excluded_market_ids_other_categories.txt"
            out_excl = os.path.abspath(out_excl)
            if "market_id" not in dropped.columns:
                raise SystemExit("Cannot write exclusions: missing market_id column.")
            ids = dropped["market_id"].astype(str).tolist()
            with open(out_excl, "w", encoding="utf-8") as f:
                for mid in ids:
                    f.write(mid + "\n")
            print(f"Wrote {len(ids)} excluded market_id(s) (other categories) -> {out_excl}")
        df = kept
        n_input = len(df)

    exclude = parse_exclude_ids(args.exclude_market_ids, args.exclude_market_ids_file)
    if exclude:
        mid = df["market_id"].astype(str)
        keep = ~mid.isin(exclude)
        df = df.loc[keep].reset_index(drop=True)
        n_excluded = int(keep.eq(False).sum())
        if n_excluded == 0:
            print("Warning: no rows matched --exclude-market-ids; check IDs match CSV market_id.", file=sys.stderr)
        else:
            print(f"Excluded {n_excluded} row(s) from PCA fit ({len(exclude)} id(s) in exclusion set).")

    if len(df) < 2:
        raise SystemExit(
            "Need at least 2 rows after exclusions for PCA. Remove some IDs from the exclusion list."
        )

    feat_cols = select_feature_columns(df)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_comp = max_n_components(len(df), X.shape[1], int(args.max_pca_components))
    pca = PCA(n_components=n_comp, random_state=0)
    Z_full = pca.fit_transform(X)
    ev_full = pca.explained_variance_ratio_
    if Z_full.shape[1] < 2:
        Z = np.column_stack([Z_full[:, 0], np.zeros(len(df), dtype=float)])
        ev = np.array([float(ev_full[0]), 0.0], dtype=float)
    else:
        Z = Z_full[:, :2].copy()
        ev = ev_full[:2].astype(float)

    print_pc_feature_correlations(X, Z_full[:, : min(2, Z_full.shape[1])], feat_cols, top_n=8)

    meta_csv = resolve_metadata_csv(path, args.metadata)
    enriched = merge_question(df, meta_csv)

    plot_df = pd.DataFrame(
        {
            "PC1": Z[:, 0],
            "PC2": Z[:, 1],
            "market_id": enriched["market_id"],
            "classifier_label": enriched["classifier_label"].astype(str),
            "tag_slug": enriched["tag_slug"].astype(str),
            "_question": enriched["_question"],
        }
    )

    title_suffix = ""
    if allowed is not None:
        shown = ", ".join(sorted({str(x) for x in df["classifier_label"].astype(str).unique()}))
        title_suffix += f"<br><sup>Categories: {shown}</sup>"
    if exclude:
        title_suffix += f"<br><sup>PCA fit on {len(df)} markets ({n_input - len(df)} excluded by id)</sup>"

    out = os.path.abspath(args.output)
    stem = os.path.splitext(out)[0]
    ext = os.path.splitext(out)[1].lower()

    if not args.no_extra_plots:
        write_scree_html(ev_full, stem + "_scree.html", title_suffix=title_suffix)
        write_loadings_html(
            pca.components_,
            feat_cols,
            stem + "_loadings.html",
            title_suffix=title_suffix,
            df_rows=df,
            Z=Z_full,
        )
        n_h = len(df)
        idx = np.arange(n_h)
        if n_h > args.heatmap_max_markets:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(n_h, size=args.heatmap_max_markets, replace=False))
            print(
                f"Distance heatmap: random subsample of {args.heatmap_max_markets} / {n_h} markets (seed=0)."
            )
        if args.distance_space == "latent":
            ddim = max(1, min(int(args.distance_pc_dims), Z_full.shape[1]))
            Zd = Z_full[idx, :ddim]
            desc = f"latent (first {ddim} PCs)"
        else:
            Zd = X[idx, :]
            desc = "raw feature vectors"
        labs = df["market_id"].astype(str).iloc[idx].tolist()
        cats = (
            df["classifier_label"].astype(str).iloc[idx].tolist()
            if "classifier_label" in df.columns
            else None
        )
        write_distance_heatmap_html(
            Zd,
            labs,
            stem + "_distances.html",
            categories=cats,
            title_suffix=title_suffix,
            space_desc=desc,
        )
        print(
            f"Wrote scree / loadings / distance heatmap: {stem}_scree.html, "
            f"{stem}_loadings.html, {stem}_distances.html"
        )

    if ext in (".html", ".htm"):
        write_plotly_html(
            plot_df,
            ev,
            out,
            title_suffix=title_suffix,
            include_outlier_panel=not args.no_outlier_panel,
        )
        print(f"Wrote interactive {out} ({len(df)} points, {len(feat_cols)} features).")
        print(f"PCA feature columns: {', '.join(feat_cols)}")
        print(
            "Open in a browser: box-zoom / scroll zoom / pan on the toolbar; double-click resets axes."
        )
        if not args.no_outlier_panel:
            print(
                "To exclude outliers: toolbar → Box Select or Lasso Select → download IDs → rerun with "
                "--exclude-market-ids-file excluded_market_ids.txt"
            )
        if meta_csv:
            print(f"Hover text includes questions from {meta_csv}")
    elif ext == ".png":
        write_matplotlib_png(plot_df, ev, out, args.dpi)
        print(f"Wrote static {out} ({len(df)} points, {len(feat_cols)} features).")
        print(f"PCA feature columns: {', '.join(feat_cols)}")
    else:
        raise SystemExit(
            f"Unknown output extension {ext!r}; use .html (interactive) or .png (static)."
        )


if __name__ == "__main__":
    main()
