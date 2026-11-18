"""Step 2 - Candidate generation + feature engineering for train and student_test.

For every trip:
  1. load + clean + project the track (pathsnap.data_io)
  2. derive observation-level reference arcs from .route (pathsnap.labels)
  3. generate candidate arcs within CANDIDATE_RADIUS_M (pathsnap.network)
  4. compute candidate features (pathsnap.features)

Outputs (results/features/):
  <partition>_candidates.parquet   one row per (trip, obs, candidate arc) with features + y
  <partition>_observations.csv     one row per GPS observation: reference arc, distance, candidate hit
  candidate_stats.csv              candidate miss rate etc. per trip

Run:  python scripts/02_build_features.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from pathsnap import config, data_io, features, labels, network

OUT = config.FEATURES_DIR
OUT.mkdir(parents=True, exist_ok=True)


def main():
    stats = []
    for part, pdir in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        tables, obs_rows = [], []
        for tid in data_io.list_trips(pdir):
            t0 = time.time()
            trip = data_io.load_trip(pdir, tid)
            lab, ldist = labels.align_route(trip)
            net = network.RoadNetwork(trip)
            df = features.build_candidate_table(trip, net, lab)
            hit = np.zeros(len(trip.xy), bool)
            hit[df.loc[df.y == 1, "obs"].values] = True
            nearest = df[df.dist_rank == 0]
            obs_rows.append(pd.DataFrame({"trip": tid, "obs": np.arange(len(trip.xy)), "ref_arc": lab,
                                          "ref_dist_m": ldist, "ref_in_candidates": hit}))
            tables.append(df)
            stats.append({"partition": part, "trip": tid, "n_obs": len(trip.xy), "n_rows": len(df),
                          "mean_candidates": len(df) / len(trip.xy),
                          "obs_without_candidates": int(len(trip.xy) - df.obs.nunique()),
                          "candidate_miss_rate": float(1 - hit.mean()),
                          "nearest_arc_top1": float(nearest.y.sum() / len(trip.xy)),
                          "label_dist_median_m": float(np.median(ldist)),
                          "seconds": round(time.time() - t0, 2)})
            print(stats[-1])
        pd.concat(tables).to_parquet(OUT / f"{part}_candidates.parquet", index=False)
        pd.concat(obs_rows).to_csv(OUT / f"{part}_observations.csv", index=False)
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "candidate_stats.csv", index=False)
    print(st.groupby("partition")[["mean_candidates", "candidate_miss_rate", "nearest_arc_top1"]].mean())


if __name__ == "__main__":
    main()
