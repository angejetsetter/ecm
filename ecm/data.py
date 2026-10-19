"""Data schema, loading/validation, and a synthetic data generator.

Two tables drive every analysis:

clients (one row per client, measured at intake)
    client_id         unique id
    age               years
    phq2              PHQ-2 depression screen, 0-6
    gad2              GAD-2 anxiety screen, 0-6
    auditc            AUDIT-C alcohol use screen, 0-12
    hrsn_<domain>     1 if the client screened positive for that health-related
                      social need (housing, food, transport, utilities, safety)
    hrsn_total        number of positive HRSN domains (derived if missing)
    engagement_30d    contacts with the program in the first 30 days
    chc               1 if the client participated in the CHC program
    last_contact_month  (optional) month of the last contact of any kind;
                      if absent it is taken from the latest visit

visits (long format, one row per client per assessment)
    client_id, month (0 = intake), phq2, gad2, auditc, hrsn_total

Only engagement from the first 30 days is used as a predictor so that the risk
model never sees information from the period it is trying to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_RANGES = {"phq2": (0, 6), "gad2": (0, 6), "auditc": (0, 12)}
HRSN_DOMAINS = ["housing", "food", "transport", "utilities", "safety"]
HRSN_COLUMNS = [f"hrsn_{d}" for d in HRSN_DOMAINS]

CLIENT_REQUIRED = ["client_id", "age", "phq2", "gad2", "auditc", "engagement_30d", "chc"]
VISIT_REQUIRED = ["client_id", "month", "phq2", "gad2", "auditc"]


def validate_clients(clients: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and score ranges; derive hrsn_total if needed."""
    clients = clients.copy()
    missing = [c for c in CLIENT_REQUIRED if c not in clients.columns]
    if missing:
        raise ValueError(f"clients table is missing columns: {missing}")
    if clients["client_id"].duplicated().any():
        raise ValueError("clients table has duplicate client_id values")
    if "hrsn_total" not in clients.columns:
        present = [c for c in HRSN_COLUMNS if c in clients.columns]
        if not present:
            raise ValueError("clients table needs hrsn_total or hrsn_<domain> columns")
        clients["hrsn_total"] = clients[present].sum(axis=1)
    for col, (lo, hi) in SCORE_RANGES.items():
        bad = clients[col].notna() & ~clients[col].between(lo, hi)
        if bad.any():
            raise ValueError(f"{col} has {int(bad.sum())} values outside {lo}-{hi}")
    return clients


def validate_visits(visits: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in VISIT_REQUIRED if c not in visits.columns]
    if missing:
        raise ValueError(f"visits table is missing columns: {missing}")
    return visits.sort_values(["client_id", "month"]).reset_index(drop=True)


def load_data(clients_path: str, visits_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients = validate_clients(pd.read_csv(clients_path))
    visits = validate_visits(pd.read_csv(visits_path))
    return clients, visits


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #

# Latent subgroups baked into the synthetic data so the clustering step has
# something real to recover. Means are on the raw score scale; HRSN entries are
# probabilities of screening positive for each domain.
_ARCHETYPES = {
    "depression_housing": dict(
        weight=0.25, age=(42, 12), phq2=4.8, gad2=3.0, auditc=2.0,
        hrsn=dict(housing=0.75, food=0.60, transport=0.50, utilities=0.50, safety=0.20)),
    "alcohol_low_needs": dict(
        weight=0.20, age=(38, 11), phq2=1.5, gad2=1.5, auditc=8.0,
        hrsn=dict(housing=0.10, food=0.10, transport=0.15, utilities=0.10, safety=0.05)),
    "anxiety_food_transport": dict(
        weight=0.20, age=(35, 10), phq2=1.8, gad2=5.0, auditc=2.0,
        hrsn=dict(housing=0.20, food=0.55, transport=0.50, utilities=0.30, safety=0.15)),
    "low_acuity": dict(
        weight=0.35, age=(50, 14), phq2=0.8, gad2=0.8, auditc=1.5,
        hrsn=dict(housing=0.08, food=0.08, transport=0.08, utilities=0.08, safety=0.03)),
}

# True CHC effect on each score, in points per month (negative = improvement).
TRUE_CHC_EFFECT_PER_MONTH = {"phq2": -0.12, "gad2": -0.10, "auditc": -0.10}

VISIT_MONTHS = (0, 3, 6, 9, 12)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _observe(latent, rng, col):
    lo, hi = SCORE_RANGES[col]
    return np.clip(np.round(latent + rng.normal(0, 0.8, size=np.shape(latent))), lo, hi)


def generate_synthetic(n_clients: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate intake data and quarterly follow-up for a CHC-style program.

    Built-in structure, so each analysis has a known answer to recover:
      * four latent subgroups (see _ARCHETYPES)
      * CHC enrolment is confounded: sicker clients with more social needs and
        higher early engagement are more likely to join
      * CHC participation lowers scores at TRUE_CHC_EFFECT_PER_MONTH
      * dropout is more likely for younger, low-engagement, heavier-drinking
        clients with more social needs
    """
    rng = np.random.default_rng(seed)
    names = list(_ARCHETYPES)
    weights = np.array([_ARCHETYPES[n]["weight"] for n in names])
    group = rng.choice(len(names), size=n_clients, p=weights / weights.sum())

    rows = {"client_id": [f"C{i:05d}" for i in range(n_clients)], "archetype": []}
    latent = {k: np.empty(n_clients) for k in SCORE_RANGES}
    age = np.empty(n_clients)
    hrsn = {d: np.empty(n_clients, dtype=int) for d in HRSN_DOMAINS}
    for g, name in enumerate(names):
        a = _ARCHETYPES[name]
        idx = group == g
        n = idx.sum()
        age[idx] = rng.normal(*a["age"], size=n)
        for k in SCORE_RANGES:
            latent[k][idx] = rng.normal(a[k], 1.0, size=n)
        for d in HRSN_DOMAINS:
            hrsn[d][idx] = rng.binomial(1, a["hrsn"][d], size=n)
    rows["archetype"] = [names[g] for g in group]
    age = np.clip(np.round(age), 18, 85)
    hrsn_total = sum(hrsn.values())

    engagement = rng.poisson(np.exp(1.1 + 0.01 * (age - 45) - 0.06 * latent["auditc"]
                                    - 0.05 * hrsn_total))

    baseline = {k: _observe(latent[k], rng, k) for k in SCORE_RANGES}
    p_chc = _sigmoid(-0.6 + 0.35 * (baseline["phq2"] - 2.5) + 0.30 * (hrsn_total - 1.5)
                     + 0.20 * (engagement - 3))
    chc = rng.binomial(1, p_chc)

    # Monthly dropout hazard -> first month with no further contact.
    hazard = _sigmoid(-3.6 - 0.025 * (age - 45) + 0.18 * (latent["auditc"] - 3)
                      - 0.30 * (engagement - 3) + 0.20 * (hrsn_total - 1.5) - 0.5 * chc)
    months = np.arange(1, max(VISIT_MONTHS) + 1)
    events = rng.random((n_clients, len(months))) < hazard[:, None]
    dropout_month = np.where(events.any(axis=1), months[events.argmax(axis=1)], np.inf)

    # Score trajectories: social needs drag scores up, CHC pulls them down.
    slopes = {}
    for k in SCORE_RANGES:
        natural = -0.02 + 0.03 * (hrsn_total - 1.5)
        slopes[k] = (natural + TRUE_CHC_EFFECT_PER_MONTH[k] * chc
                     + rng.normal(0, 0.04, size=n_clients))

    visit_rows = []
    for m in VISIT_MONTHS:
        seen = dropout_month > m
        obs = {k: (baseline[k] if m == 0 else _observe(latent[k] + slopes[k] * m, rng, k))
               for k in SCORE_RANGES}
        hrsn_m = np.clip(hrsn_total - (rng.random(n_clients) < 0.04 * m * (0.5 + chc)), 0, 5)
        if m == 0:
            hrsn_m = hrsn_total
        visit_rows.append(pd.DataFrame({
            "client_id": np.array(rows["client_id"])[seen],
            "month": m,
            **{k: v[seen].astype(int) for k, v in obs.items()},
            "hrsn_total": np.asarray(hrsn_m)[seen].astype(int),
        }))
    visits = pd.concat(visit_rows, ignore_index=True)

    last_contact = np.where(np.isinf(dropout_month), max(VISIT_MONTHS), dropout_month - 1)
    clients = pd.DataFrame({
        "client_id": rows["client_id"],
        "age": age.astype(int),
        **{k: v.astype(int) for k, v in baseline.items()},
        **{f"hrsn_{d}": v for d, v in hrsn.items()},
        "hrsn_total": hrsn_total.astype(int),
        "engagement_30d": engagement.astype(int),
        "chc": chc.astype(int),
        "last_contact_month": last_contact.astype(int),
        "archetype": rows["archetype"],
    })
    return validate_clients(clients), validate_visits(visits)
