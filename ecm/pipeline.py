"""Run risk prediction, client grouping and outcome analysis end to end.

    python -m ecm.pipeline --synthetic --out output/
    python -m ecm.pipeline --clients clients.csv --visits visits.csv --out output/
    python -m ecm.pipeline --clients clients.csv --visits visits.csv \
        --new-clients intake_this_week.csv --out output/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .clustering import cluster_clients
from .data import generate_synthetic, load_data, validate_clients
from .outcomes import estimate_chc_effects
from .risk import derive_outcomes, train_risk_model


def _md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    df = df.copy()
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(floatfmt)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def run(clients: pd.DataFrame, visits: pd.DataFrame, out_dir: Path,
        new_clients: pd.DataFrame | None = None, horizon: int = 6, k: int | None = None,
        seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Risk prediction
    outcomes = derive_outcomes(clients, visits, horizon=horizon)
    risk = train_risk_model(clients, outcomes, seed=seed)
    risk.metrics.to_csv(out_dir / "risk_model_metrics.csv", index=False)

    # 2. Client groups
    groups = cluster_clients(clients, k=k, seed=seed)
    assignments = pd.DataFrame({"client_id": clients["client_id"], "cluster": groups.labels,
                                "cluster_name": [groups.names[c] for c in groups.labels]})
    assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
    groups.profiles.to_csv(out_dir / "cluster_profiles.csv", index=False)
    # Flags carry each client's group so outreach can be matched to it.
    to_score = new_clients if new_clients is not None else clients
    flags = risk.flag(to_score).merge(groups.assign(to_score)[["client_id", "cluster_name"]],
                                      on="client_id")
    flags.to_csv(out_dir / "risk_flags.csv", index=False)
    # Outcome rates by group tie the grouping back to the risk question.
    by_group = (assignments.merge(outcomes, on="client_id").groupby("cluster_name")
                [["dropped_out", "worsened", "high_risk"]].mean().round(3).reset_index())

    # 3. Outcomes
    effects = estimate_chc_effects(clients, visits, at_month=horizon)
    effects.effects.to_csv(out_dir / "chc_effects.csv", index=False)
    effects.balance.to_csv(out_dir / "covariate_balance.csv", index=False)

    best = risk.metrics.iloc[0]
    report = f"""# Client risk, grouping and outcomes report

Clients: {len(clients)} · Visits: {len(visits)} · Horizon: {horizon} months

## 1. Who is likely to get worse or drop out?

Outcome = dropped out before month {horizon} **or** PHQ-2 / GAD-2 / AUDIT-C rose by
>= 2 points at the month-{horizon} follow-up. {int(outcomes['high_risk'].notna().sum())} clients had a
known outcome; {outcomes['high_risk'].mean():.1%} were high-risk.

Cross-validated model comparison (best model refit on all data and used for flags):

{_md(risk.metrics)}

Selected: **{risk.model_name}**. At the flag threshold ({best['threshold']:.3f}) the model
catches {best['sensitivity']:.0%} of clients who went on to worsen or drop out while
flagging {best['share_flagged']:.0%} of all clients (PPV {best['ppv']:.0%}).

What drives risk (logistic regression, odds ratio per 1 SD increase):

{_md(risk.odds_ratios)}

Permutation importance for the selected model (drop in AUC when shuffled):

{_md(risk.importance)}

Flags for {len(flags)} {'new ' if new_clients is not None else ''}clients: `risk_flags.csv`
({int(flags['flagged'].sum())} flagged high-risk).

## 2. Client groups

K-means on intake PHQ-2, GAD-2, AUDIT-C and number of unmet social needs; k = {groups.k} chosen by silhouette.

{_md(groups.silhouette_by_k)}

{_md(groups.profiles, 2)}

6-month outcomes by group:

{_md(by_group)}

## 3. Did CHC participation improve scores?

Effect = difference in score at month {horizon} between CHC participants and comparable
non-participants (negative = CHC participants improved more). The naive row is
unadjusted and shown only for contrast.

{_md(effects.effects.drop(columns=['random_effects'], errors='ignore'))}

Covariate balance (standardised mean difference, CHC vs non-CHC; target |SMD| < 0.1 after weighting):

{_md(effects.balance)}

These estimates adjust only for measured intake differences. Unmeasured
differences between people who chose to join CHC and those who did not can
still bias them.
"""
    (out_dir / "report.md").write_text(report)
    return {"outcomes": outcomes, "risk": risk, "flags": flags, "groups": groups,
            "group_outcomes": by_group, "effects": effects, "report": report}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clients")
    ap.add_argument("--visits")
    ap.add_argument("--new-clients", help="intake data for clients to score")
    ap.add_argument("--synthetic", action="store_true", help="use simulated data")
    ap.add_argument("--n", type=int, default=1500, help="synthetic client count")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--k", type=int, help="fix the number of client groups")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.synthetic:
        clients, visits = generate_synthetic(args.n, seed=args.seed)
        out.mkdir(parents=True, exist_ok=True)
        clients.to_csv(out / "synthetic_clients.csv", index=False)
        visits.to_csv(out / "synthetic_visits.csv", index=False)
    elif args.clients and args.visits:
        clients, visits = load_data(args.clients, args.visits)
    else:
        ap.error("pass --synthetic or both --clients and --visits")
    new = validate_clients(pd.read_csv(args.new_clients)) if args.new_clients else None

    result = run(clients, visits, out, new_clients=new, horizon=args.horizon, k=args.k,
                 seed=args.seed)
    print(result["report"])
    print(f"\nWrote outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
