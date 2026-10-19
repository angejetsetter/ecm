import numpy as np
import pandas as pd
import pytest

from ecm.clustering import cluster_clients
from ecm.data import generate_synthetic, validate_clients
from ecm.outcomes import estimate_chc_effects
from ecm.pipeline import run
from ecm.risk import derive_outcomes, train_risk_model


@pytest.fixture(scope="module")
def data():
    return generate_synthetic(1200, seed=1)


def test_synthetic_data_is_valid(data):
    clients, visits = data
    assert clients["client_id"].is_unique
    assert set(visits["month"]) == {0, 3, 6, 9, 12}
    assert visits["phq2"].between(0, 6).all() and visits["auditc"].between(0, 12).all()
    # Dropout means fewer visits at later months.
    counts = visits.groupby("month").size()
    assert counts[12] < counts[0]


def test_validate_rejects_out_of_range_scores(data):
    bad = data[0].copy()
    bad.loc[0, "phq2"] = 9
    with pytest.raises(ValueError, match="phq2"):
        validate_clients(bad)


def test_derive_outcomes():
    clients = pd.DataFrame({"client_id": ["a", "b", "c", "d"], "phq2": [2, 2, 2, 2],
                            "gad2": [1, 1, 1, 1], "auditc": [3, 3, 3, 3],
                            "last_contact_month": [12, 12, 2, 12]})
    visits = pd.DataFrame({"client_id": ["a", "b"], "month": [6, 6], "phq2": [4, 2],
                           "gad2": [1, 1], "auditc": [3, 4]})
    out = derive_outcomes(clients, visits).set_index("client_id")
    assert out.loc["a", "high_risk"] == 1     # PHQ-2 rose by 2
    assert out.loc["b", "high_risk"] == 0     # AUDIT-C rose by only 1
    assert out.loc["c", "high_risk"] == 1     # dropped out at month 2
    assert np.isnan(out.loc["d", "high_risk"])  # engaged but no 6-month score


def test_risk_model_beats_chance_and_hits_sensitivity(data):
    clients, visits = data
    risk = train_risk_model(clients, derive_outcomes(clients, visits))
    best = risk.metrics.iloc[0]
    assert best["auc"] > 0.6
    assert best["sensitivity"] == pytest.approx(0.8, abs=0.02)
    flags = risk.flag(clients.head(50))
    assert set(flags["risk_tier"]) <= {"high", "medium", "low"}
    assert flags["risk_score"].is_monotonic_decreasing
    # Low early engagement should raise risk in the synthetic data.
    odds = risk.odds_ratios.set_index("feature")["odds_ratio_per_sd"]
    assert odds["engagement_30d"] < 1


def test_clustering_recovers_synthetic_subgroups(data):
    clients, _ = data
    res = cluster_clients(clients)
    assert res.k == 4
    purity = pd.crosstab(res.labels, clients["archetype"]).max(axis=1).sum() / len(clients)
    assert purity > 0.85
    names = " | ".join(res.names.values())
    assert "high AUDIT-C + low social needs" in names
    assert "housing" in names
    assigned = res.assign(clients.head(10))
    assert (assigned["cluster"].to_numpy() == res.labels[:10]).all()


def test_chc_effects_are_adjusted_and_negative(data):
    clients, visits = data
    res = estimate_chc_effects(clients, visits)
    eff = res.effects.set_index(["score", "method"])
    for s in ("phq2", "gad2", "auditc"):
        for m in ("mixed_model", "ipw_ancova"):
            assert eff.loc[(s, m), "ci_high"] < 0, (s, m)
    assert (res.balance["smd_weighted"].abs() < 0.1).all()
    assert (res.balance["smd_unweighted"].abs().max() > 0.3)


def test_pipeline_writes_outputs(data, tmp_path):
    clients, visits = data
    result = run(clients, visits, tmp_path, new_clients=clients.head(20))
    for f in ("risk_flags.csv", "risk_model_metrics.csv", "cluster_assignments.csv",
              "cluster_profiles.csv", "chc_effects.csv", "covariate_balance.csv", "report.md"):
        assert (tmp_path / f).exists(), f
    flags = pd.read_csv(tmp_path / "risk_flags.csv")
    assert len(flags) == 20 and "cluster_name" in flags
