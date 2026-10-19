# ECM client analytics

Three analyses on intake screening data (PHQ-2, GAD-2, AUDIT-C, HRSN) plus
follow-up assessments, for a CHC-style care management program.

| Question | Module | Method |
|---|---|---|
| Who is likely to get worse or drop out within 6 months? | `ecm/risk.py` | Logistic regression vs. random forest, 5-fold CV; best model flags new clients at intake |
| What natural subgroups exist? | `ecm/clustering.py` | K-means on standardised scores + HRSN count, k by silhouette, auto-named groups |
| Did CHC participation improve scores? | `ecm/outcomes.py` | Longitudinal mixed model + propensity-weighted ANCOVA, with covariate balance checks |

## Quick start

```bash
pip install -r requirements.txt

# Try it on simulated data (known answers built in)
python -m ecm.pipeline --synthetic --out output/

# Your data
python -m ecm.pipeline --clients clients.csv --visits visits.csv --out output/

# Train on past clients, flag this week's intakes
python -m ecm.pipeline --clients clients.csv --visits visits.csv \
    --new-clients new_intakes.csv --out output/
```

Outputs in `output/`: `report.md` (readable summary), `risk_flags.csv`,
`risk_model_metrics.csv`, `cluster_assignments.csv`, `cluster_profiles.csv`,
`chc_effects.csv`, `covariate_balance.csv`.

## Input data

**clients.csv** — one row per client, at intake

| column | meaning |
|---|---|
| `client_id` | unique id |
| `age` | years |
| `phq2`, `gad2`, `auditc` | intake scores (0–6, 0–6, 0–12) |
| `hrsn_housing`, `hrsn_food`, `hrsn_transport`, `hrsn_utilities`, `hrsn_safety` | 1 = screened positive for that need |
| `hrsn_total` | number of positive domains (derived from the flags if missing) |
| `engagement_30d` | program contacts in the **first 30 days only** |
| `chc` | 1 = participated in CHC |
| `last_contact_month` | optional; month of last contact (otherwise taken from visits) |

**visits.csv** — one row per assessment: `client_id, month, phq2, gad2, auditc[, hrsn_total]`,
with `month = 0` at intake.

## Definitions and design choices

**High-risk outcome.** A client is high-risk if they dropped out (last contact
before month 5) **or** at the follow-up nearest month 6 (months 5–7) any of
PHQ-2, GAD-2 or AUDIT-C had risen by ≥ 2 points. Clients still engaged but
missing that assessment are excluded from training rather than guessed at.
Thresholds live in `risk.WORSENING_THRESHOLDS`.

**No leakage.** Predictors are intake values plus engagement in the first 30
days, so a client can be scored at day 30 using only what is known then.

**Flag threshold.** Chosen on out-of-fold predictions to catch ~80% of clients
who went on to worsen or drop out (`target_sensitivity`). The report shows the
trade-off: what share of clients get flagged and how many flags are true
positives (PPV). Lower the target if outreach capacity is tight.

**Explaining risk.** Odds ratios per 1 SD from the logistic model give
direction (e.g. more unmet social needs → higher risk; more early engagement →
lower risk); permutation importance shows what the selected model relies on.

**Groups.** HRSN enters clustering as the count of unmet needs; individual
domains are used to describe each group ("high PHQ-2 + housing & food
insecurity"). Use `--k` to fix the number of groups. The report shows
dropout/worsening rates per group so programs can be tailored to each one.
New clients are assigned to the existing groups in `risk_flags.csv`.

**Outcomes.** CHC participants usually differ from non-participants at intake
(in the simulated data they are sicker and have more social needs), so a raw
before/after comparison is misleading. Two adjusted estimates are reported:

1. *Mixed model* — every follow-up score, adjusted for that client's baseline
   score and intake characteristics, with a random intercept and slope per
   client: `score_t ~ baseline + chc * month + age + hrsn_total + engagement_30d + …`.
   Handles missed visits if they are missing at random.
2. *IPW ANCOVA* — clients are weighted by inverse probability of joining CHC
   so the groups match at intake (check `covariate_balance.csv`, target
   |SMD| < 0.1), then the month-6 score is regressed on CHC + covariates.

Agreement between the two is reassuring. Both adjust only for **measured**
differences; motivation or other unrecorded factors can still bias them.
A negative effect means CHC participants ended with lower (better) scores than
comparable non-participants.

## Synthetic data

`ecm.data.generate_synthetic()` simulates four subgroups (depression + housing
insecurity, alcohol + low social needs, anxiety + food/transport needs, low
acuity), confounded CHC enrolment, a true CHC effect
(`TRUE_CHC_EFFECT_PER_MONTH`), and dropout driven by age, engagement, alcohol
use and social needs. The tests check that each analysis recovers this
structure. Observed effects are somewhat smaller than the true latent effect
because scores are bounded (a client at PHQ-2 = 0 cannot improve).

## Tests

```bash
python -m pytest
```
