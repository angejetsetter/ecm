"""Find natural subgroups of clients from their intake profile.

K-means on standardised intake scores and HRSN domains, with the number of
groups chosen by silhouette score. Each group gets a plain-language label built
from the features where it sits furthest from the overall client average, e.g.
"high PHQ-2 + housing need" or "high AUDIT-C + low social needs".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import HRSN_COLUMNS

SCORE_FEATURES = ["phq2", "gad2", "auditc"]
# HRSN enters as the count of unmet needs: binary domain flags distort k-means
# distances, so individual domains are used for describing groups instead.
CLUSTER_FEATURES = SCORE_FEATURES + ["hrsn_total"]

_PRETTY = {"phq2": "PHQ-2 (depression)", "gad2": "GAD-2 (anxiety)",
           "auditc": "AUDIT-C (alcohol)", "hrsn_total": "social needs", "age": "age",
           "engagement_30d": "engagement"}


@dataclass
class ClusterResult:
    labels: np.ndarray
    k: int
    silhouette_by_k: pd.DataFrame
    profiles: pd.DataFrame
    names: dict[int, str]
    model: KMeans
    scaler: StandardScaler
    features: list[str]

    def assign(self, clients: pd.DataFrame) -> pd.DataFrame:
        """Assign new clients to the existing groups."""
        cid = self.model.predict(self.scaler.transform(clients[self.features].to_numpy(dtype=float)))
        return pd.DataFrame({"client_id": clients["client_id"].to_numpy(), "cluster": cid,
                             "cluster_name": [self.names[c] for c in cid]})


def _name_cluster(z: pd.Series, cutoff: float = 0.5) -> str:
    """Label a group by where it sits >= `cutoff` SDs from the client average."""
    parts = [f"high {_PRETTY[f].split()[0]}" for f in SCORE_FEATURES if z[f] >= cutoff]
    needs = [c.removeprefix("hrsn_") for c in HRSN_COLUMNS if c in z and z[c] >= cutoff]
    needs = sorted(needs, key=lambda d: -z[f"hrsn_{d}"])[:2]
    low_needs = "hrsn_total" in z and z["hrsn_total"] <= -cutoff
    if not parts and all(z[f] <= 0 for f in SCORE_FEATURES):
        parts.append("low acuity")
    if needs:
        parts.append(" & ".join(needs) + " insecurity")
    elif low_needs and parts != ["low acuity"]:
        parts.append("low social needs")
    return " + ".join(parts) or "mixed / average"


def cluster_clients(clients: pd.DataFrame, features: list[str] = CLUSTER_FEATURES,
                    k_range: range = range(2, 9), k: int | None = None,
                    seed: int = 0) -> ClusterResult:
    """Cluster clients; `k` fixes the number of groups, otherwise it is the
    value in `k_range` with the best silhouette score."""
    features = [f for f in features if f in clients.columns]
    X = clients[features].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    sil = []
    for kk in k_range:
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Xs)
        sample = min(len(Xs), 3000)
        sil.append({"k": kk, "silhouette": silhouette_score(Xs, km.labels_, sample_size=sample,
                                                           random_state=seed)})
    sil = pd.DataFrame(sil)
    if k is None:
        k = int(sil.loc[sil["silhouette"].idxmax(), "k"])
    model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Xs)
    labels = model.labels_

    profile_cols = list(dict.fromkeys(features + HRSN_COLUMNS
                                      + ["hrsn_total", "age", "engagement_30d"]))
    profile_cols = [c for c in profile_cols if c in clients.columns]
    df = clients[profile_cols].assign(cluster=labels)
    means = df.groupby("cluster")[profile_cols].mean()
    overall_mean, overall_sd = clients[profile_cols].mean(), clients[profile_cols].std(ddof=0)
    z = (means - overall_mean) / overall_sd.replace(0, 1)
    names = {c: _name_cluster(z.loc[c]) for c in means.index}

    profiles = means.round(2)
    profiles.insert(0, "n", df.groupby("cluster").size())
    profiles.insert(1, "share", (profiles["n"] / len(df)).round(3))
    profiles.insert(0, "name", pd.Series(names))
    for extra in ("chc",):
        if extra in clients.columns:
            profiles[f"{extra}_rate"] = clients.groupby(labels)[extra].mean().round(3)

    return ClusterResult(labels=labels, k=k, silhouette_by_k=sil, profiles=profiles.reset_index(),
                         names=names, model=model, scaler=scaler, features=features)
