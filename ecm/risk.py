"""Predict which clients are likely to get worse or drop out within 6 months.

Outcome (per past client):
    dropped_out  last contact before month `horizon - window`
    worsened     at the follow-up closest to `horizon` (within +/- window
                 months), any score rose by at least its threshold
    high_risk    dropped_out OR worsened

Clients still engaged but with no follow-up assessment have an unknown outcome
and are left out of training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RISK_FEATURES = ["phq2", "gad2", "auditc", "hrsn_total", "age", "engagement_30d"]

# Smallest increase treated as clinically meaningful worsening.
WORSENING_THRESHOLDS = {"phq2": 2, "gad2": 2, "auditc": 2}


def derive_outcomes(clients: pd.DataFrame, visits: pd.DataFrame, horizon: int = 6,
                    window: int = 1,
                    thresholds: dict[str, int] = WORSENING_THRESHOLDS) -> pd.DataFrame:
    """Label each client with dropped_out / worsened / high_risk (NaN if unknown)."""
    last_contact = (clients.set_index("client_id")["last_contact_month"]
                    if "last_contact_month" in clients.columns
                    else visits.groupby("client_id")["month"].max())
    out = pd.DataFrame(index=pd.Index(clients["client_id"], name="client_id"))
    out["dropped_out"] = (last_contact.reindex(out.index).fillna(0) < horizon - window).astype(int)

    fu = visits[visits["month"].between(horizon - window, horizon + window)].copy()
    fu["dist"] = (fu["month"] - horizon).abs()
    fu = fu.sort_values(["client_id", "dist"]).drop_duplicates("client_id").set_index("client_id")
    base = clients.set_index("client_id")

    rises = pd.concat([(fu[s] - base[s].reindex(fu.index)) >= t for s, t in thresholds.items()],
                      axis=1)
    worsened = pd.Series(np.nan, index=out.index)
    worsened[fu.index] = rises.any(axis=1).astype(int)
    out["worsened"] = worsened

    high = worsened.copy()
    high[out["dropped_out"] == 1] = 1
    out["high_risk"] = high
    return out.reset_index()


@dataclass
class RiskModel:
    model: object
    model_name: str
    features: list[str]
    threshold: float
    metrics: pd.DataFrame
    importance: pd.DataFrame
    odds_ratios: pd.DataFrame | None = None
    tier_cutoffs: tuple[float, float] = field(default=(0.0, 0.0))

    def predict(self, clients: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(clients[self.features])[:, 1]

    def flag(self, clients: pd.DataFrame) -> pd.DataFrame:
        """Score new clients at intake. Tiers: high = at/above the flag threshold."""
        p = self.predict(clients)
        medium_cut = self.tier_cutoffs[0]
        tier = np.where(p >= self.threshold, "high", np.where(p >= medium_cut, "medium", "low"))
        out = clients[["client_id"]].copy()
        out["risk_score"] = p.round(3)
        out["risk_tier"] = tier
        out["flagged"] = (p >= self.threshold).astype(int)
        return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


def _candidates(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=15, class_weight="balanced",
            random_state=seed, n_jobs=-1),
    }


def _threshold_for_sensitivity(y, p, target: float) -> float:
    """Highest cut-off that still catches `target` share of true high-risk clients."""
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[max(k, 0)])


def train_risk_model(clients: pd.DataFrame, outcomes: pd.DataFrame,
                     features: list[str] = RISK_FEATURES, target_sensitivity: float = 0.80,
                     n_splits: int = 5, seed: int = 0) -> RiskModel:
    """Compare logistic regression and random forest with cross-validation,
    keep the one with the best out-of-fold AUC, and refit it on all data.

    The flag threshold is chosen on out-of-fold predictions so that roughly
    `target_sensitivity` of clients who went on to worsen or drop out would
    have been flagged at intake.
    """
    df = clients.merge(outcomes[["client_id", "high_risk"]], on="client_id")
    df = df.dropna(subset=features + ["high_risk"])
    X, y = df[features], df["high_risk"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows, oof = [], {}
    for name, est in _candidates(seed).items():
        p = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
        oof[name] = p
        rows.append({"model": name, "n": len(y), "event_rate": y.mean(),
                     "auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p),
                     "brier": brier_score_loss(y, p)})
    metrics = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    best = metrics.loc[0, "model"]

    p_best = oof[best]
    threshold = _threshold_for_sensitivity(y, p_best, target_sensitivity)
    flagged = p_best >= threshold
    metrics.loc[0, "threshold"] = threshold
    metrics.loc[0, "sensitivity"] = flagged[y == 1].mean()
    metrics.loc[0, "specificity"] = (~flagged[y == 0]).mean()
    metrics.loc[0, "ppv"] = y[flagged].mean() if flagged.any() else np.nan
    metrics.loc[0, "share_flagged"] = flagged.mean()

    model = _candidates(seed)[best].fit(X, y)
    imp = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10, random_state=seed)
    importance = (pd.DataFrame({"feature": features, "auc_drop": imp.importances_mean,
                                "sd": imp.importances_std})
                  .sort_values("auc_drop", ascending=False).reset_index(drop=True))

    # Odds ratios per 1 SD are always reported from the logistic model because
    # they are the easiest way to explain direction of effect to program staff.
    lr = _candidates(seed)["logistic_regression"].fit(X, y)
    coefs = lr[-1].coef_[0]
    odds = pd.DataFrame({"feature": features, "odds_ratio_per_sd": np.exp(coefs),
                         "sd": lr[0].scale_}).sort_values("odds_ratio_per_sd", ascending=False)

    return RiskModel(model=model, model_name=best, features=list(features), threshold=threshold,
                     metrics=metrics, importance=importance, odds_ratios=odds.reset_index(drop=True),
                     tier_cutoffs=(float(np.quantile(p_best, 0.5)), threshold))
