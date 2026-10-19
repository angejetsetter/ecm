"""Did CHC participation improve clients' scores, after accounting for who joined?

Two complementary estimates per score:

1. Longitudinal mixed model (uses every follow-up visit, tolerates missed
   visits under a missing-at-random assumption):

       score_t ~ baseline_score + chc * month + age + hrsn_total + engagement_30d
                 + (1 + month | client)

   The CHC effect at month m is  b_chc + m * b_chc:month.

2. Propensity-weighted ANCOVA at a single follow-up month: the probability of
   joining CHC is modelled from intake characteristics, clients are weighted by
   stabilised inverse-probability weights so the two groups look alike at
   intake, and the follow-up score is regressed on CHC plus the same
   covariates (doubly robust). Covariate balance before/after weighting is
   reported so the adjustment can be checked.

Both only adjust for differences that were measured. Negative effects mean CHC
participants ended with lower (better) scores than comparable non-participants.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCORES = ["phq2", "gad2", "auditc"]
CONFOUNDERS = ["phq2", "gad2", "auditc", "hrsn_total", "age", "engagement_30d"]


@dataclass
class OutcomeResult:
    effects: pd.DataFrame
    balance: pd.DataFrame
    mixed_models: dict


def _long_with_baseline(clients: pd.DataFrame, visits: pd.DataFrame, score: str) -> pd.DataFrame:
    covs = [c for c in CONFOUNDERS if c != score]
    base = clients[["client_id", "chc", score] + covs].rename(columns={score: "baseline"})
    long = visits.loc[visits["month"] > 0, ["client_id", "month", score]].merge(base, on="client_id")
    return long.dropna(), covs


def mixed_model_effect(clients, visits, score: str, at_month: int = 6):
    long, covs = _long_with_baseline(clients, visits, score)
    centred = {c: long[c] - long[c].mean() for c in ["baseline"] + covs}
    long = long.assign(**{f"{c}_c": v for c, v in centred.items()})
    formula = (f"{score} ~ baseline_c + chc * month + "
               + " + ".join(f"{c}_c" for c in covs))
    fit = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for re in ("~month", "~1"):
            try:
                fit = smf.mixedlm(formula, long, groups=long["client_id"], re_formula=re).fit(
                    method="lbfgs", reml=True)
                if fit.converged:
                    break
            except (np.linalg.LinAlgError, ValueError):
                continue
    if fit is None:
        raise RuntimeError(f"mixed model failed for {score}")

    L = pd.Series(0.0, index=fit.fe_params.index)
    L["chc"], L["chc:month"] = 1.0, float(at_month)
    est = float(L @ fit.fe_params)
    se = float(np.sqrt(L @ fit.cov_params().loc[L.index, L.index] @ L))
    return fit, {
        "score": score, "method": "mixed_model", "month": at_month,
        "effect": est, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
        "p_value": float(2 * (1 - norm.cdf(abs(est / se)))),
        "slope_diff_per_month": float(fit.fe_params["chc:month"]),
        "n_clients": long["client_id"].nunique(), "n_obs": len(long),
        "random_effects": re,
    }


def propensity_weights(clients: pd.DataFrame, confounders=CONFOUNDERS,
                       trim: tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """Stabilised inverse-probability-of-participation weights."""
    X, t = clients[confounders], clients["chc"].to_numpy()
    ps = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, t)
    p = np.clip(ps.predict_proba(X)[:, 1], *trim)
    pt = t.mean()
    w = np.where(t == 1, pt / p, (1 - pt) / (1 - p))
    return pd.Series(w, index=clients.index, name="ipw")


def balance_table(clients: pd.DataFrame, weights: pd.Series, confounders=CONFOUNDERS):
    """Standardised mean differences; |SMD| < 0.1 is the usual target."""
    t = clients["chc"] == 1
    rows = []
    for c in confounders:
        x = clients[c]
        pooled_sd = np.sqrt((x[t].var() + x[~t].var()) / 2) or 1.0
        raw = (x[t].mean() - x[~t].mean()) / pooled_sd
        wm1 = np.average(x[t], weights=weights[t])
        wm0 = np.average(x[~t], weights=weights[~t])
        rows.append({"covariate": c, "smd_unweighted": raw, "smd_weighted": (wm1 - wm0) / pooled_sd})
    return pd.DataFrame(rows)


def ipw_ancova_effect(clients, visits, score: str, weights: pd.Series, at_month: int = 6):
    fu = visits.loc[visits["month"] == at_month, ["client_id", score]].rename(
        columns={score: "followup"})
    df = clients.assign(ipw=weights).merge(fu, on="client_id").dropna(
        subset=["followup"] + CONFOUNDERS)
    X = sm.add_constant(df[["chc"] + CONFOUNDERS].astype(float))
    fit = sm.WLS(df["followup"], X, weights=df["ipw"]).fit(cov_type="HC3")
    est, (lo, hi) = fit.params["chc"], fit.conf_int().loc["chc"]
    return {"score": score, "method": "ipw_ancova", "month": at_month, "effect": float(est),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": float(fit.pvalues["chc"]),
            "n_clients": len(df), "n_obs": len(df)}


def naive_change(clients, visits, score: str, at_month: int = 6) -> dict:
    """Unadjusted difference in mean change, shown for contrast only."""
    fu = visits.loc[visits["month"] == at_month, ["client_id", score]]
    df = clients[["client_id", "chc", score]].merge(fu, on="client_id", suffixes=("_0", "_f"))
    df["change"] = df[f"{score}_f"] - df[f"{score}_0"]
    g = df.groupby("chc")["change"]
    m, s, n = g.mean(), g.std(), g.size()
    diff = m.get(1, np.nan) - m.get(0, np.nan)
    se = np.sqrt((s**2 / n).sum())
    return {"score": score, "method": "naive_change", "month": at_month, "effect": diff,
            "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se,
            "p_value": float(2 * (1 - norm.cdf(abs(diff / se)))),
            "n_clients": len(df), "n_obs": len(df)}


def estimate_chc_effects(clients: pd.DataFrame, visits: pd.DataFrame, at_month: int = 6,
                         scores=SCORES) -> OutcomeResult:
    w = propensity_weights(clients)
    rows, fits = [], {}
    for s in scores:
        fit, row = mixed_model_effect(clients, visits, s, at_month)
        fits[s] = fit
        rows += [naive_change(clients, visits, s, at_month), row,
                 ipw_ancova_effect(clients, visits, s, w, at_month)]
    effects = pd.DataFrame(rows)
    return OutcomeResult(effects=effects, balance=balance_table(clients, w), mixed_models=fits)
