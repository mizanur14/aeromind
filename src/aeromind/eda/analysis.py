"""Reusable exploratory-analysis functions for C-MAPSS training trajectories.

All functions are descriptive: they summarise a given frame and fit nothing that is later
used by models. Which engines the frame contains (full training pool vs training split)
is decided by the caller; see :mod:`aeromind.eda.report`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from aeromind.features.conditions import OperatingConditionModel
from aeromind.features.targets import cap_rul


def lifetime_table(df: pd.DataFrame) -> pd.DataFrame:
    """Engine lifetimes (number of cycles; failure at the last cycle)."""
    return df.groupby("engine_id")["cycle"].max().rename("lifetime").reset_index()


def sensor_statistics(df: pd.DataFrame, columns: list[str], tol: float = 1e-8) -> pd.DataFrame:
    """Per-column descriptive statistics with constant / near-constant flags.

    ``near_constant`` marks non-constant columns with at most 3 distinct values (e.g. a
    quantised channel toggling between two readings). A coefficient-of-variation rule is
    deliberately not used: it misflags large-offset signals such as fan speeds near
    2388 rpm whose small drift is informative.
    """
    values = df[columns]
    out = values.agg(["mean", "std", "min", "max"]).T
    out["range"] = out["max"] - out["min"]
    out["unique_values"] = values.nunique()
    out["missing"] = values.isna().sum()
    out["cv"] = out["std"] / out["mean"].abs().replace(0, np.nan)
    out["constant"] = out["range"] <= tol
    out["near_constant"] = ~out["constant"] & (out["unique_values"] <= 3)
    return out


def rul_correlations(df: pd.DataFrame, columns: list[str], cap: int = 125) -> pd.DataFrame:
    """Spearman correlation of each column with uncapped and capped RUL (row level).

    Row-level correlations weight long engines more and are associations, not causes.
    """
    rul = df["rul"].to_numpy(dtype=float)
    capped = cap_rul(rul, cap)
    rows = []
    for col in columns:
        x = df[col].to_numpy(dtype=float)
        if np.ptp(x) == 0:
            rows.append({"column": col, "spearman_rul": np.nan, "spearman_rul_capped": np.nan})
            continue
        rows.append(
            {
                "column": col,
                "spearman_rul": sps.spearmanr(x, rul).statistic,
                "spearman_rul_capped": sps.spearmanr(x, capped).statistic,
            }
        )
    out = pd.DataFrame(rows)
    return out.reindex(out["spearman_rul"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame, columns: list[str], method: str = "spearman") -> pd.DataFrame:
    """Pairwise correlation matrix of non-constant columns."""
    usable = [c for c in columns if df[c].nunique() > 1]
    return df[usable].corr(method=method)


def within_engine_trends(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Monotonic degradation trend per engine: Spearman(sensor, cycle) within each engine.

    Returns per-column median, IQR and the fraction of engines whose trend has the same
    sign as the median (consistency of the degradation direction).
    """
    rows = []
    groups = list(df.groupby("engine_id"))
    for col in columns:
        rhos = []
        for _, g in groups:
            x = g[col].to_numpy(dtype=float)
            rhos.append(np.nan if np.ptp(x) == 0 else sps.spearmanr(g["cycle"], x).statistic)
        r = np.asarray(rhos, dtype=float)
        valid = r[np.isfinite(r)]
        if valid.size == 0:
            rows.append({"column": col, "median_rho": np.nan, "iqr_rho": np.nan, "sign_consistency": np.nan})
            continue
        med = float(np.median(valid))
        rows.append(
            {
                "column": col,
                "median_rho": med,
                "iqr_rho": float(np.subtract(*np.percentile(valid, [75, 25]))),
                "sign_consistency": float(np.mean(np.sign(valid) == np.sign(med))),
            }
        )
    return pd.DataFrame(rows).sort_values("median_rho", key=np.abs, ascending=False).reset_index(drop=True)


def between_engine_variability(df: pd.DataFrame, columns: list[str], k: int = 20) -> pd.DataFrame:
    """Compare engine-to-engine spread with within-engine noise.

    For each column: engine means over the first ``k`` cycles (healthy) and last ``k``
    cycles (pre-failure). ``between_sd_early`` is the across-engine SD of healthy means,
    ``noise_sd`` the pooled within-engine SD in the first ``k`` cycles, and
    ``degradation_shift_sd`` the mean (late − early) shift in noise-SD units.
    """
    head = df.groupby("engine_id").head(k)
    tail = df.groupby("engine_id").tail(k)
    rows = []
    for col in columns:
        early = head.groupby("engine_id")[col].mean()
        late = tail.groupby("engine_id")[col].mean()
        noise = float(np.sqrt(head.groupby("engine_id")[col].var(ddof=1).mean()))
        noise = noise if noise > 0 else np.nan
        rows.append(
            {
                "column": col,
                "between_sd_early": float(early.std(ddof=1)),
                "between_sd_late": float(late.std(ddof=1)),
                "noise_sd": noise,
                "between_to_noise_early": float(early.std(ddof=1)) / noise if noise else np.nan,
                "degradation_shift_sd": float((late - early).mean()) / noise if noise else np.nan,
            }
        )
    return pd.DataFrame(rows)


def operating_conditions(df: pd.DataFrame, n_conditions: int, seed: int = 42) -> pd.DataFrame:
    """Cluster settings into operating conditions; per-condition row counts and setting means."""
    model = OperatingConditionModel(n_conditions, seed).fit(df)
    cond = model.predict(df)
    settings = ["setting_1", "setting_2", "setting_3"]
    frame = df[settings].assign(condition=cond)
    out = frame.groupby("condition")[settings].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out.insert(0, "rows", frame.groupby("condition").size())
    return out.reset_index()


def early_late_comparison(df: pd.DataFrame, columns: list[str], frac: float = 0.2) -> pd.DataFrame:
    """Two-sample KS test between the first and last ``frac`` of each engine's life."""
    pos = df.groupby("engine_id").cumcount()
    n = df.groupby("engine_id")["cycle"].transform("size")
    early = df[pos < frac * n]
    late = df[pos >= (1 - frac) * n]
    rows = []
    for col in columns:
        a, b = early[col].to_numpy(float), late[col].to_numpy(float)
        if np.ptp(np.r_[a, b]) == 0:
            rows.append({"column": col, "ks_stat": 0.0, "ks_pvalue": 1.0, "mean_shift_sd": 0.0})
            continue
        ks = sps.ks_2samp(a, b)
        sd = a.std(ddof=1) or np.nan
        rows.append(
            {
                "column": col,
                "ks_stat": ks.statistic,
                "ks_pvalue": ks.pvalue,
                "mean_shift_sd": (b.mean() - a.mean()) / sd,
            }
        )
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)


def outlier_summary(df: pd.DataFrame, columns: list[str], z: float = 5.0) -> pd.DataFrame:
    """Count robust outliers after removing each engine's quadratic degradation trend.

    Residuals from a per-engine quadratic fit in cycle are scored with a robust z-score
    (median/MAD per column). This is descriptive only; nothing is removed from the data.
    """
    resid = pd.DataFrame(index=df.index, columns=columns, dtype=float)
    for _, g in df.groupby("engine_id"):
        t = g["cycle"].to_numpy(float)
        deg = 2 if len(g) > 3 else 0
        for col in columns:
            y = g[col].to_numpy(float)
            coef = np.polyfit(t, y, deg)
            resid.loc[g.index, col] = y - np.polyval(coef, t)
    rows = []
    for col in columns:
        r = resid[col].to_numpy(float)
        mad = np.median(np.abs(r - np.median(r))) * 1.4826
        if mad <= 0:
            rows.append({"column": col, "outliers": 0, "fraction": 0.0, "engines_affected": 0})
            continue
        mask = np.abs((r - np.median(r)) / mad) > z
        rows.append(
            {
                "column": col,
                "outliers": int(mask.sum()),
                "fraction": float(mask.mean()),
                "engines_affected": int(df.loc[mask, "engine_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def variance_inflation(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """VIF of each non-constant column: diagonal of the inverse (Pearson) correlation matrix."""
    corr = correlation_matrix(df, columns, method="pearson")
    inv = np.linalg.pinv(corr.to_numpy())
    return (
        pd.DataFrame({"column": corr.columns, "vif": np.diag(inv)})
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Column pairs with ``|r| >= threshold`` (upper triangle)."""
    cols = corr.columns
    rows = [
        {"a": cols[i], "b": cols[j], "r": float(corr.iat[i, j])}
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
        if abs(corr.iat[i, j]) >= threshold
    ]
    return pd.DataFrame(rows, columns=["a", "b", "r"]).sort_values("r", key=np.abs, ascending=False)
