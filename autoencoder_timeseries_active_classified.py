#!/usr/bin/env python3
"""
Deep non-linear and semi-supervised autoencoders for active-classified Polymarket time series.

**Unsupervised (strong non-linear AE)**  
Deeper tanh encoder / decoder with linear reconstruction head, gradient clipping, and
weight clipping for stability.

**Semi-supervised**  
Same encoder + decoder, plus a linear classifier on the latent:
`loss = MSE(recon, x) + alpha * CE(softmax(z @ Wc + bc), category)`.

**Conditional VAE (CVAE)**  
Encoder sees `concat(x, y_onehot)`; decoder sees `concat(z, y_onehot)` with reparameterized
`z = mu + exp(0.5 log sigma^2) * eps`. Loss is `MSE(x_hat, x) + beta * KL(q(z|x,y) || N(0,I))`.
Plots use **mu** (mean latent) for stable 2D visualization.

Each market is one sequence: pivot `price_yes` on `lookback_step` (or `normalized_step`),
concatenate `[prices | first-differences]`, standardize, then train.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def pick_step_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"--step-column {explicit!r} not found in raw CSV.")
        return explicit
    for c in ("lookback_step", "normalized_step"):
        if c in df.columns:
            return c
    raise SystemExit("Could not find step column; expected lookback_step or normalized_step.")


def build_timeseries_matrix(
    raw: pd.DataFrame,
    *,
    step_col: str,
    min_points_per_market: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"market_id", "classifier_label", "tag_slug", "price_yes", step_col}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise SystemExit(f"Raw CSV missing required columns: {missing}")

    df = raw.copy()
    df = df[df["market_id"].notna()].copy()
    df["market_id"] = df["market_id"].astype(str)
    df["price_yes"] = pd.to_numeric(df["price_yes"], errors="coerce")
    df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
    df = df[df[step_col].notna()].copy()

    sort_cols = ["market_id", step_col]
    if "timestamp_iso" in df.columns:
        sort_cols.append("timestamp_iso")
    df = df.sort_values(sort_cols)
    dedup = df.drop_duplicates(subset=["market_id", step_col], keep="last")

    piv = dedup.pivot(index="market_id", columns=step_col, values="price_yes")
    piv = piv.sort_index(axis=1)

    counts = piv.notna().sum(axis=1)
    keep = counts >= int(min_points_per_market)
    piv = piv.loc[keep].copy()
    if piv.empty:
        raise SystemExit(
            f"No markets remain after min_points_per_market={min_points_per_market} filter."
        )

    piv = piv.ffill(axis=1).bfill(axis=1).fillna(0.5)

    meta_cols = ["market_id", "classifier_label", "tag_slug"]
    if "question" in dedup.columns:
        meta_cols.append("question")
    meta = (
        dedup[meta_cols]
        .drop_duplicates(subset=["market_id"], keep="last")
        .set_index("market_id")
        .reindex(piv.index)
        .reset_index()
    )
    if "question" not in meta.columns:
        meta["question"] = ""
    meta["question"] = meta["question"].astype(str).fillna("")
    meta["classifier_label"] = meta["classifier_label"].astype(str).fillna("Unknown")
    meta["tag_slug"] = meta["tag_slug"].astype(str).fillna("")
    return piv, meta


def build_rich_features(seq_df: pd.DataFrame) -> np.ndarray:
    x_price = seq_df.to_numpy(dtype=float)
    x_diff = np.diff(x_price, axis=1, prepend=x_price[:, :1])
    return np.concatenate([x_price, x_diff], axis=1)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -20.0, 20.0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    z = np.clip(z, -30.0, 30.0)
    e = np.exp(z)
    s = e.sum(axis=1, keepdims=True) + 1e-12
    return e / s


def _clip_grads(arrays: List[Optional[np.ndarray]], clip: float = 5.0) -> float:
    g2 = sum(float(np.sum(g * g)) for g in arrays if g is not None)
    gnorm = float(np.sqrt(max(g2, 1e-12)))
    return min(1.0, clip / gnorm)


def train_deep_autoencoder(
    X: np.ndarray,
    *,
    h1: int,
    h2: int,
    latent_dim: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    batch_size: int,
    random_state: int,
    y_onehot: Optional[np.ndarray] = None,
    semi_weight: float = 0.0,
) -> Tuple[np.ndarray, dict]:
    """
    Encoder: x -> tanh -> tanh -> z=tanh (bottleneck)
    Decoder: z -> tanh -> tanh -> x_hat (linear last layer)
    Optional: logits = z @ Wc + bc, joint loss with semi_weight * CE.
    """
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    zd = int(latent_dim)
    use_semi = y_onehot is not None and float(semi_weight) > 0.0
    n_classes = int(y_onehot.shape[1]) if use_semi else 0

    scale = 0.08 / max(np.sqrt(float(d)), 1.0)
    w1 = rng.normal(0.0, scale, size=(d, h1))
    b1 = np.zeros((1, h1))
    w2 = rng.normal(0.0, scale, size=(h1, h2))
    b2 = np.zeros((1, h2))
    wz = rng.normal(0.0, scale, size=(h2, zd))
    bz = np.zeros((1, zd))
    w3 = rng.normal(0.0, scale, size=(zd, h2))
    b3 = np.zeros((1, h2))
    w4 = rng.normal(0.0, scale, size=(h2, h1))
    b4 = np.zeros((1, h1))
    w5 = rng.normal(0.0, scale, size=(h1, d))
    b5 = np.zeros((1, d))
    wc = np.zeros((zd, n_classes)) if use_semi else None
    bc = np.zeros((1, n_classes)) if use_semi else None
    if use_semi and wc is not None:
        wc[:] = rng.normal(0.0, 0.02, size=wc.shape)

    bs = max(8, int(batch_size))
    lr = float(learning_rate)
    alpha = float(semi_weight)

    for _ in range(max(1, int(epochs))):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            idx = order[s : s + bs]
            xb = X[idx]
            m = float(len(xb))
            yb = y_onehot[idx] if use_semi else None

            gwc: Optional[np.ndarray] = None
            gbc: Optional[np.ndarray] = None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                a1 = _tanh(xb @ w1 + b1)
                a2 = _tanh(a1 @ w2 + b2)
                z = _tanh(a2 @ wz + bz)
                d1 = _tanh(z @ w3 + b3)
                d2 = _tanh(d1 @ w4 + b4)
                xh = d2 @ w5 + b5

                err = xh - xb
                dxh = 2.0 * err / max(m, 1.0)

                # Decoder backward
                gw5 = d2.T @ dxh + l2 * w5
                gb5 = dxh.sum(axis=0, keepdims=True)
                dd2 = dxh @ w5.T
                du4 = dd2 * (1.0 - d2 * d2)
                gw4 = d1.T @ du4 + l2 * w4
                gb4 = du4.sum(axis=0, keepdims=True)
                dd1 = du4 @ w4.T
                du3 = dd1 * (1.0 - d1 * d1)
                gw3 = z.T @ du3 + l2 * w3
                gb3 = du3.sum(axis=0, keepdims=True)
                dz_dec = du3 @ w3.T

                dz = dz_dec.copy()
                if use_semi and wc is not None and bc is not None and yb is not None:
                    logits = z @ wc + bc
                    p = _softmax(logits)
                    dlog = (p - yb) / max(m, 1.0)
                    gwc = z.T @ dlog + l2 * wc
                    gbc = dlog.sum(axis=0, keepdims=True)
                    dz_ce = dlog @ wc.T
                    dz = dz_dec + alpha * dz_ce

                duz = dz * (1.0 - z * z)
                gwz = a2.T @ duz + l2 * wz
                gbz = duz.sum(axis=0, keepdims=True)
                da2 = duz @ wz.T
                du2 = da2 * (1.0 - a2 * a2)
                gw2 = a1.T @ du2 + l2 * w2
                gb2 = du2.sum(axis=0, keepdims=True)
                da1 = du2 @ w2.T
                du1 = da1 * (1.0 - a1 * a1)
                gw1 = xb.T @ du1 + l2 * w1
                gb1 = du1.sum(axis=0, keepdims=True)

            for g in (gw1, gw2, gwz, gw3, gw4, gw5, gwc):
                if g is not None:
                    np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            for g in (gb1, gb2, gbz, gb3, gb4, gb5, gbc):
                if g is not None:
                    np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            gs = _clip_grads([gw1, gw2, gwz, gw3, gw4, gw5, gwc], clip=5.0)
            w1 -= lr * gs * gw1
            b1 -= lr * gs * gb1
            w2 -= lr * gs * gw2
            b2 -= lr * gs * gb2
            wz -= lr * gs * gwz
            bz -= lr * gs * gbz
            w3 -= lr * gs * gw3
            b3 -= lr * gs * gb3
            w4 -= lr * gs * gw4
            b4 -= lr * gs * gb4
            w5 -= lr * gs * gw5
            b5 -= lr * gs * gb5
            if use_semi and gwc is not None and gbc is not None and wc is not None and bc is not None:
                wc -= lr * gs * gwc
                bc -= lr * gs * gbc

            np.clip(w1, -4.0, 4.0, out=w1)
            np.clip(w2, -4.0, 4.0, out=w2)
            np.clip(wz, -4.0, 4.0, out=wz)
            np.clip(w3, -4.0, 4.0, out=w3)
            np.clip(w4, -4.0, 4.0, out=w4)
            np.clip(w5, -4.0, 4.0, out=w5)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a1 = _tanh(X @ w1 + b1)
        a2 = _tanh(a1 @ w2 + b2)
        z_all = _tanh(a2 @ wz + bz)
    z_all = np.nan_to_num(z_all, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a1f = _tanh(X @ w1 + b1)
        a2f = _tanh(a1f @ w2 + b2)
        zf = _tanh(a2f @ wz + bz)
        d1f = _tanh(zf @ w3 + b3)
        d2f = _tanh(d1f @ w4 + b4)
        xhf = d2f @ w5 + b5
    final_mse = float(np.mean((xhf - X) ** 2))

    params = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "wz": wz,
        "bz": bz,
        "w3": w3,
        "b3": b3,
        "w4": w4,
        "b4": b4,
        "w5": w5,
        "b5": b5,
        "wc": wc,
        "bc": bc,
        "final_mse": final_mse,
    }
    return z_all, params


def train_conditional_vae(
    X: np.ndarray,
    y_onehot: np.ndarray,
    *,
    h1: int,
    h2: int,
    latent_dim: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    batch_size: int,
    beta_kl: float,
    random_state: int,
) -> Tuple[np.ndarray, dict]:
    """
    Conditional VAE: q(z|x,y), p(x|z,y). Encoder and decoder both concatenate one-hot labels.
    Returns mean latent mu for each row (for plotting / export) plus diagnostics.
    """
    n, d_x = X.shape
    n_c = int(y_onehot.shape[1])
    if n_c < 2:
        raise SystemExit("CVAE needs at least 2 label classes.")
    rng = np.random.default_rng(random_state)
    zd = int(latent_dim)
    enc_in = d_x + n_c
    dec_in = zd + n_c

    scale = 0.06 / max(np.sqrt(float(enc_in)), 1.0)
    # Encoder
    w1 = rng.normal(0.0, scale, size=(enc_in, h1))
    b1 = np.zeros((1, h1))
    w2 = rng.normal(0.0, scale, size=(h1, h2))
    b2 = np.zeros((1, h2))
    w_mu = rng.normal(0.0, scale, size=(h2, zd))
    b_mu = np.zeros((1, zd))
    w_lv = rng.normal(0.0, scale, size=(h2, zd))
    b_lv = np.zeros((1, zd))
    # Decoder p(x|z,y)
    wd1 = rng.normal(0.0, scale, size=(dec_in, h2))
    bd1 = np.zeros((1, h2))
    wd2 = rng.normal(0.0, scale, size=(h2, h1))
    bd2 = np.zeros((1, h1))
    wd3 = rng.normal(0.0, scale, size=(h1, d_x))
    bd3 = np.zeros((1, d_x))

    bs = max(8, int(batch_size))
    lr = float(learning_rate)
    beta = float(beta_kl)

    for _ in range(max(1, int(epochs))):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            idx = order[s : s + bs]
            xb = X[idx]
            yb = y_onehot[idx]
            m = float(len(xb))
            xc = np.concatenate([xb, yb], axis=1)

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                h1a = _tanh(xc @ w1 + b1)
                h2a = _tanh(h1a @ w2 + b2)
                mu = h2a @ w_mu + b_mu
                logvar = np.clip(h2a @ w_lv + b_lv, -8.0, 8.0)
                eps = rng.standard_normal(size=mu.shape)
                std = np.exp(0.5 * logvar)
                z_s = mu + std * eps
                zy = np.concatenate([z_s, yb], axis=1)
                d1 = _tanh(zy @ wd1 + bd1)
                d2 = _tanh(d1 @ wd2 + bd2)
                xh = d2 @ wd3 + bd3

                err = xh - xb
                dxh = 2.0 * err / max(m, 1.0)

                # Decoder backward
                gwd3 = d2.T @ dxh + l2 * wd3
                gbd3 = dxh.sum(axis=0, keepdims=True)
                dd2 = dxh @ wd3.T
                du2 = dd2 * (1.0 - d2 * d2)
                gwd2 = d1.T @ du2 + l2 * wd2
                gbd2 = du2.sum(axis=0, keepdims=True)
                dd1 = du2 @ wd2.T
                du1 = dd1 * (1.0 - d1 * d1)
                gwd1 = zy.T @ du1 + l2 * wd1
                gbd1 = du1.sum(axis=0, keepdims=True)
                dzy = du1 @ wd1.T
                dz = dzy[:, :zd]

                # KL(q||N(0,I)) per sample, summed over latent dims
                # KL_k = -0.5 * (1 + logvar_k - mu_k^2 - exp(logvar_k))
                dkl_mu = beta * (mu / max(m, 1.0))  # d(beta * batch-mean KL) / d mu
                dkl_lv = beta * (-0.5 * (1.0 - np.exp(logvar)) / max(m, 1.0))

                dmu = dz + dkl_mu
                dlv = dz * (0.5 * std * eps) + dkl_lv

                g_w_mu = h2a.T @ dmu + l2 * w_mu
                g_b_mu = dmu.sum(axis=0, keepdims=True)
                g_w_lv = h2a.T @ dlv + l2 * w_lv
                g_b_lv = dlv.sum(axis=0, keepdims=True)
                dh2 = dmu @ w_mu.T + dlv @ w_lv.T
                du_h2 = dh2 * (1.0 - h2a * h2a)
                g_w2 = h1a.T @ du_h2 + l2 * w2
                g_b2 = du_h2.sum(axis=0, keepdims=True)
                dh1 = du_h2 @ w2.T
                du_h1 = dh1 * (1.0 - h1a * h1a)
                g_w1 = xc.T @ du_h1 + l2 * w1
                g_b1 = du_h1.sum(axis=0, keepdims=True)

            for g in (g_w1, g_w2, g_w_mu, g_w_lv, gwd1, gwd2, gwd3):
                np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            for g in (g_b1, g_b2, g_b_mu, g_b_lv, gbd1, gbd2, gbd3):
                np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            gs = _clip_grads([g_w1, g_w2, g_w_mu, g_w_lv, gwd1, gwd2, gwd3], clip=5.0)
            w1 -= lr * gs * g_w1
            b1 -= lr * gs * g_b1
            w2 -= lr * gs * g_w2
            b2 -= lr * gs * g_b2
            w_mu -= lr * gs * g_w_mu
            b_mu -= lr * gs * g_b_mu
            w_lv -= lr * gs * g_w_lv
            b_lv -= lr * gs * g_b_lv
            wd1 -= lr * gs * gwd1
            bd1 -= lr * gs * gbd1
            wd2 -= lr * gs * gwd2
            bd2 -= lr * gs * gbd2
            wd3 -= lr * gs * gwd3
            bd3 -= lr * gs * gbd3

            np.clip(w1, -4.0, 4.0, out=w1)
            np.clip(w2, -4.0, 4.0, out=w2)
            np.clip(w_mu, -4.0, 4.0, out=w_mu)
            np.clip(w_lv, -4.0, 4.0, out=w_lv)
            np.clip(wd1, -4.0, 4.0, out=wd1)
            np.clip(wd2, -4.0, 4.0, out=wd2)
            np.clip(wd3, -4.0, 4.0, out=wd3)

    # Final mu and recon MSE for reporting
    xc_all = np.concatenate([X, y_onehot], axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h1a = _tanh(xc_all @ w1 + b1)
        h2a = _tanh(h1a @ w2 + b2)
        mu_all = h2a @ w_mu + b_mu
        logv_all = np.clip(h2a @ w_lv + b_lv, -8.0, 8.0)
        z0 = mu_all
        zy0 = np.concatenate([z0, y_onehot], axis=1)
        d1f = _tanh(zy0 @ wd1 + bd1)
        d2f = _tanh(d1f @ wd2 + bd2)
        xhf = d2f @ wd3 + bd3
    final_mse = float(np.mean((xhf - X) ** 2))
    kl_report = float(
        np.mean(-0.5 * np.sum(1.0 + logv_all - mu_all * mu_all - np.exp(logv_all), axis=1))
    )
    mu_all = np.nan_to_num(mu_all, nan=0.0, posinf=0.0, neginf=0.0)
    params = {
        "final_mse": final_mse,
        "final_kl_mean": kl_report,
        "beta_kl": beta,
    }
    return mu_all, params


def write_plot(
    emb: pd.DataFrame,
    out_html: str,
    *,
    step_col: str,
    n_steps: int,
    n_features: int,
    latent_dim: int,
    subtitle: str,
) -> None:
    try:
        import plotly.express as px
    except ImportError as e:  # pragma: no cover
        print("Install plotly: pip install plotly", file=sys.stderr)
        raise SystemExit(1) from e

    xcol = "z1"
    ycol = "z2" if "z2" in emb.columns else None
    if ycol is None:
        rng = np.random.default_rng(0)
        emb = emb.copy()
        emb["_jitter"] = rng.normal(0.0, 0.01, len(emb))
        ycol = "_jitter"

    fig = px.scatter(
        emb,
        x=xcol,
        y=ycol,
        color="classifier_label",
        hover_name="market_id",
        custom_data=["tag_slug", "question"],
        title=(
            "Latent space — active-classified markets"
            f"<br><sup>{subtitle}; input={n_features} ([price|diff]), steps={n_steps}, step={step_col}, "
            f"latent_dim={latent_dim}</sup>"
        ),
        labels={xcol: "Latent dim 1", ycol: "Latent dim 2"},
    )
    fig.update_traces(
        marker=dict(size=9, line=dict(width=0.5, color="rgba(0,0,0,0.4)")),
        hovertemplate=(
            "<b>market_id</b> %{hovertext}<br>"
            "<b>classifier</b> %{fullData.name}<br>"
            "<b>tag_slug</b> %{customdata[0]}<br>"
            "<b>question</b> %{customdata[1]}<extra></extra>"
        ),
    )
    fig.update_layout(dragmode="zoom", hovermode="closest")
    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        config={"scrollZoom": True, "displaylogo": False, "displayModeBar": True},
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deep non-linear and semi-supervised autoencoders on price time series.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  unsupervised     — deep tanh autoencoder (reconstruction only)
  semi-supervised  — same + linear classifier on latent (recon + alpha * CE)
  cvae             — conditional VAE (label concat on enc/dec; ELBO + beta*KL)
  both             — unsupervised + semi-supervised (default)
  all              — all three models

Example:
  python autoencoder_timeseries_active_classified.py --mode all --epochs 1500 --semi-weight 0.8 --beta-kl 0.5
""",
    )
    ap.add_argument(
        "--raw-input",
        default="polymarket_active_classified_raw_30m.csv",
        help="Raw active-classified CSV with per-bucket prices.",
    )
    ap.add_argument(
        "--mode",
        choices=("unsupervised", "semi-supervised", "cvae", "both", "all"),
        default="both",
        help="Which model(s) to train and export.",
    )
    ap.add_argument(
        "--output-html",
        default="polymarket_active_classified_autoencoder_latent.html",
        help="Output HTML for unsupervised (or only) model.",
    )
    ap.add_argument(
        "--embeddings-csv",
        default="polymarket_active_classified_autoencoder_embeddings.csv",
        help="Embeddings CSV for unsupervised (or only) model.",
    )
    ap.add_argument(
        "--output-html-semi",
        default="polymarket_active_classified_autoencoder_latent_semi.html",
        help="Output HTML for semi-supervised model (mode both or semi-supervised).",
    )
    ap.add_argument(
        "--embeddings-csv-semi",
        default="polymarket_active_classified_autoencoder_embeddings_semi.csv",
        help="Embeddings CSV for semi-supervised model.",
    )
    ap.add_argument(
        "--output-html-cvae",
        default="polymarket_active_classified_cvae_latent.html",
        help="Output HTML for conditional VAE (mean mu).",
    )
    ap.add_argument(
        "--embeddings-csv-cvae",
        default="polymarket_active_classified_cvae_embeddings.csv",
        help="Embeddings CSV for conditional VAE.",
    )
    ap.add_argument(
        "--step-column",
        default=None,
        help="Sequence axis column (default: lookback_step, else normalized_step).",
    )
    ap.add_argument("--min-points-per-market", type=int, default=8)
    ap.add_argument("--encoder-h1", type=int, default=128, help="First encoder hidden width.")
    ap.add_argument("--encoder-h2", type=int, default=64, help="Second encoder hidden width.")
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--epochs-semi", type=int, default=None, help="Epochs for semi model; default = --epochs.")
    ap.add_argument(
        "--epochs-cvae",
        type=int,
        default=None,
        help="Epochs for CVAE; default = --epochs.",
    )
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--semi-weight",
        type=float,
        default=0.75,
        help="Weight on cross-entropy vs reconstruction for semi-supervised training.",
    )
    ap.add_argument(
        "--beta-kl",
        type=float,
        default=0.5,
        help="KL weight in CVAE loss: MSE + beta_kl * KL(q(z|x,y) || N(0,I)).",
    )
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    raw_path = os.path.abspath(args.raw_input)
    if not os.path.isfile(raw_path):
        raise SystemExit(f"Raw input not found: {raw_path}")
    raw = pd.read_csv(raw_path)

    step_col = pick_step_column(raw, args.step_column)
    seq_df, meta = build_timeseries_matrix(
        raw,
        step_col=step_col,
        min_points_per_market=args.min_points_per_market,
    )

    X = build_rich_features(seq_df)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = np.clip(Xs, -6.0, 6.0)

    if args.latent_dim < 1:
        raise SystemExit("--latent-dim must be >= 1")
    if args.encoder_h1 < args.latent_dim or args.encoder_h2 < args.latent_dim:
        raise SystemExit("--encoder-h1 and --encoder-h2 should be >= latent_dim for a meaningful bottleneck.")

    le = LabelEncoder()
    y_idx = le.fit_transform(meta["classifier_label"].astype(str).values)
    n_classes = len(le.classes_)
    y_onehot = np.zeros((len(y_idx), n_classes), dtype=float)
    y_onehot[np.arange(len(y_idx)), y_idx] = 1.0

    epochs_semi = int(args.epochs_semi) if args.epochs_semi is not None else int(args.epochs)
    epochs_cvae = int(args.epochs_cvae) if args.epochs_cvae is not None else int(args.epochs)

    def run_one(
        *,
        mode_name: str,
        use_semi: bool,
        epochs: int,
        html_path: str,
        csv_path: str,
    ) -> None:
        Z, params = train_deep_autoencoder(
            Xs,
            h1=int(args.encoder_h1),
            h2=int(args.encoder_h2),
            latent_dim=args.latent_dim,
            epochs=epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            batch_size=args.batch_size,
            random_state=args.random_state,
            y_onehot=y_onehot if use_semi else None,
            semi_weight=args.semi_weight if use_semi else 0.0,
        )
        emb = meta.copy()
        for i in range(Z.shape[1]):
            emb[f"z{i+1}"] = Z[:, i]
        if use_semi:
            emb["label_class_index"] = y_idx
            emb["label_class"] = meta["classifier_label"].astype(str).values

        csv_abs = os.path.abspath(csv_path)
        emb.to_csv(csv_abs, index=False)
        html_abs = os.path.abspath(html_path)
        sub = (
            "deep tanh autoencoder (reconstruction)"
            if not use_semi
            else f"semi-supervised AE (MSE + {args.semi_weight:g} * CE on category)"
        )
        write_plot(
            emb,
            html_abs,
            step_col=step_col,
            n_steps=seq_df.shape[1],
            n_features=X.shape[1],
            latent_dim=args.latent_dim,
            subtitle=sub,
        )
        extra = ""
        if "final_kl_mean" in params:
            extra = f"; mean KL ≈ {params['final_kl_mean']:.6f}"
        print(
            f"[{mode_name}] Wrote {html_abs} and {csv_abs} "
            f"({len(emb)} markets; final train MSE ≈ {params.get('final_mse', float('nan')):.6f}{extra})."
        )

    def run_cvae(epochs: int, html_path: str, csv_path: str) -> None:
        mu, params = train_conditional_vae(
            Xs,
            y_onehot,
            h1=int(args.encoder_h1),
            h2=int(args.encoder_h2),
            latent_dim=args.latent_dim,
            epochs=epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            batch_size=args.batch_size,
            beta_kl=args.beta_kl,
            random_state=args.random_state,
        )
        emb = meta.copy()
        for i in range(mu.shape[1]):
            emb[f"z{i+1}"] = mu[:, i]
        emb["label_class_index"] = y_idx
        emb["label_class"] = meta["classifier_label"].astype(str).values
        emb["latent_is_cvae_mu"] = 1

        csv_abs = os.path.abspath(csv_path)
        emb.to_csv(csv_abs, index=False)
        html_abs = os.path.abspath(html_path)
        sub = f"conditional VAE (y in enc/dec; beta_kl={args.beta_kl:g}; plot uses mu)"
        write_plot(
            emb,
            html_abs,
            step_col=step_col,
            n_steps=seq_df.shape[1],
            n_features=X.shape[1],
            latent_dim=args.latent_dim,
            subtitle=sub,
        )
        print(
            f"[cvae] Wrote {html_abs} and {csv_abs} "
            f"({len(emb)} markets; MSE ≈ {params.get('final_mse', float('nan')):.6f}; "
            f"mean KL ≈ {params.get('final_kl_mean', float('nan')):.6f})."
        )

    if args.mode in ("unsupervised", "both", "all"):
        run_one(
            mode_name="unsupervised",
            use_semi=False,
            epochs=int(args.epochs),
            html_path=args.output_html,
            csv_path=args.embeddings_csv,
        )
    if args.mode in ("semi-supervised", "both", "all"):
        if n_classes < 2:
            print("Warning: fewer than 2 classes; semi-supervised head is weak or undefined.", file=sys.stderr)
        run_one(
            mode_name="semi-supervised",
            use_semi=True,
            epochs=epochs_semi,
            html_path=args.output_html_semi,
            csv_path=args.embeddings_csv_semi,
        )

    if args.mode in ("cvae", "all"):
        if n_classes < 2:
            raise SystemExit("CVAE requires at least 2 distinct classifier_label values.")
        run_cvae(epochs_cvae, args.output_html_cvae, args.embeddings_csv_cvae)

    print(
        f"Done. markets={len(meta)}, seq_len={seq_df.shape[1]}, features={X.shape[1]}, "
        f"classes={n_classes}, mode={args.mode}."
    )


if __name__ == "__main__":
    main()
