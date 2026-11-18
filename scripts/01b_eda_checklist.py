"""Step 1b - Dataset-exploration checklist (Dataset Exploration.pdf, section 4).

Answers each checklist item with numbers computed from the RAW files of the
training and student-testing partitions (evaluator data is not used):
  1 trips per partition            7 geographical extent
  2 GPS observations per trip      8 road-network nodes / segments
  3 file availability and sizes    9 road-network connectivity
  4 data types and column names   10 isolated nodes / disconnected components
  5 missing / invalid GPS         11 large GPS gaps and other anomalies
  6 timestamp order and gaps      12 reference-route availability and structure

Outputs (results/eda/checklist/): per-trip CSVs, checklist_summary.json, plots.
Run after 01_eda.py:  python scripts/01b_eda_checklist.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import config, data_io, geo

OUT = config.RESULTS_DIR / "eda" / "checklist"
OUT.mkdir(parents=True, exist_ok=True)
EXT = ["track", "nodes", "arcs", "route"]
EXPECTED_COLS = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}


def read_raw(path):
    """Read a whitespace file as text tokens, return (array of floats, n_cols set, all_integer_valued)."""
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    ncols = sorted({len(r) for r in rows})
    arr = np.array([[float(x) for x in r] for r in rows]) if len(ncols) == 1 else None
    return arr, ncols, len(rows)


def main():
    trips, types = [], []
    for part, pdir in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        for tid in data_io.list_trips(pdir):
            d = pdir / tid
            rec = {"partition": part, "trip": tid}
            arrs = {}
            for e in EXT:
                f = d / f"{tid}.{e}"
                rec[f"has_{e}"] = f.exists()
                rec[f"kb_{e}"] = round(f.stat().st_size / 1024, 1) if f.exists() else np.nan
                if f.exists():
                    a, ncols, nrows = read_raw(f)
                    arrs[e] = a.reshape(-1, 1) if (a is not None and a.ndim == 1) else a
                    rec[f"rows_{e}"] = nrows
                    rec[f"cols_{e}"] = ",".join(map(str, ncols))
                    integer_valued = bool(a is not None and np.all(np.mod(a, 1) == 0))
                    types.append({"partition": part, "trip": tid, "file": e, "columns": ",".join(map(str, ncols)),
                                  "integer_valued": integer_valued})
            tr, nd, ar, rt = arrs["track"], arrs["nodes"], arrs["arcs"].astype(np.int64), arrs["route"][:, 0].astype(np.int64)
            lon, lat, t = tr[:, 0], tr[:, 1], tr[:, 2]
            dt = np.diff(t)
            # 5. missing / invalid
            rec["n_nan"] = int(np.sum(~np.isfinite(tr)))
            rec["n_out_of_range"] = int(np.sum((np.abs(lon) > 180) | (np.abs(lat) > 90)))
            rec["n_zero_coords"] = int(np.sum((lon == 0) & (lat == 0)))
            rec["n_duplicate_rows"] = int(len(tr) - len(np.unique(tr, axis=0)))
            rec["n_repeated_position"] = int(np.sum((np.diff(lon) == 0) & (np.diff(lat) == 0)))
            nb_lo, nb_hi = nd.min(0), nd.max(0)
            rec["n_gps_outside_network_bbox"] = int(np.sum((lon < nb_lo[0]) | (lon > nb_hi[0]) | (lat < nb_lo[1]) | (lat > nb_hi[1])))
            # 6. timestamps
            rec["t_start"] = float(t[0]); rec["t_end"] = float(t[-1])
            rec["t_integer"] = bool(np.all(np.mod(t, 1) == 0))
            rec["n_non_increasing"] = int(np.sum(dt <= 0))
            rec["frac_dt_1s"] = float(np.mean(dt == 1))
            rec["n_gaps_gt1s"] = int(np.sum(dt > 1)); rec["n_gaps_gt5s"] = int(np.sum(dt > 5))
            rec["n_gaps_gt30s"] = int(np.sum(dt > 30)); rec["n_gaps_gt300s"] = int(np.sum(dt > 300))
            rec["max_gap_s"] = float(dt.max()); rec["duration_s"] = float(t[-1] - t[0])
            # 7. extent
            proj = geo.LocalProjection(lon.mean(), lat.mean())
            xy = proj.forward(lon, lat)
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            speed = step / np.maximum(dt, 1e-9)
            rec.update(lon_min=lon.min(), lon_max=lon.max(), lat_min=lat.min(), lat_max=lat.max(),
                       extent_ew_km=(xy[:, 0].max() - xy[:, 0].min()) / 1000,
                       extent_ns_km=(xy[:, 1].max() - xy[:, 1].min()) / 1000,
                       path_len_km=step.sum() / 1000, median_speed_kmh=float(np.median(speed) * 3.6),
                       n_speed_gt250kmh=int(np.sum(speed > 250 / 3.6)), max_speed_kmh=float(speed.max() * 3.6), max_jump_m=float(step.max()))
            # 8-10. network
            V, E = len(nd), len(ar)
            rec["n_nodes"], rec["n_arcs"] = V, E
            rec["n_arcs_bad_node_id"] = int(np.sum((ar < 0) | (ar >= V)))
            ok = np.all((ar >= 0) & (ar < V), axis=1)
            a2 = ar[ok]
            G = csr_matrix((np.ones(len(a2)), (a2[:, 0], a2[:, 1])), shape=(V, V))
            nw, lw = connected_components(G, directed=True, connection="weak")
            ns, ls = connected_components(G, directed=True, connection="strong")
            deg = np.bincount(a2[:, 0], minlength=V) + np.bincount(a2[:, 1], minlength=V)
            sizes_w = np.bincount(lw)
            rec.update(n_isolated_nodes=int(np.sum(deg == 0)), n_weak_components=int(nw),
                       n_weak_components_size_gt1=int(np.sum(sizes_w > 1)),
                       largest_weak_frac=float(sizes_w.max() / V), n_strong_components=int(ns),
                       largest_strong_frac=float(np.bincount(ls).max() / V),
                       n_self_loops=int(np.sum(a2[:, 0] == a2[:, 1])),
                       n_duplicate_arcs=int(len(a2) - len(np.unique(a2, axis=0))),
                       n_duplicate_nodes=int(V - len(np.unique(nd, axis=0))),
                       mean_out_degree=float(len(a2) / V))
            key = set(map(tuple, a2.tolist()))
            rec["twoway_frac"] = float(np.mean([(v, u) in key for u, v in a2.tolist()]))
            seg = np.linalg.norm(proj.forward(nd[a2[:, 1], 0], nd[a2[:, 1], 1]) - proj.forward(nd[a2[:, 0], 0], nd[a2[:, 0], 1]), axis=1)
            rec["seg_len_median_m"] = float(np.median(seg)); rec["seg_len_max_m"] = float(seg.max())
            # 12. route
            rec["route_len"] = len(rt); rec["route_unique"] = int(len(np.unique(rt)))
            rec["route_ids_valid"] = bool(np.all((rt >= 0) & (rt < E)))
            rec["route_connected_frac"] = float(np.mean(ar[rt[:-1], 1] == ar[rt[1:], 0])) if len(rt) > 1 else 1.0
            rec["route_repeated_arcs"] = int(len(rt) - len(np.unique(rt)))
            rec["route_in_largest_strong"] = bool(np.all(ls[ar[rt, 0]] == np.argmax(np.bincount(ls))))
            rec["obs_per_route_arc"] = len(tr) / max(len(rt), 1)
            rec["n_obs"] = len(tr)
            trips.append(rec)
            print(part, tid, "ok", flush=True)

    df = pd.DataFrame(trips)
    lq = pd.read_csv(config.RESULTS_DIR / "eda" / "label_quality.csv", dtype={"trip": str})
    df = df.merge(lq[["trip", "label_dist_median_m"]], on="trip", how="left")
    df.to_csv(OUT / "checklist_per_trip.csv", index=False)
    pd.DataFrame(types).to_csv(OUT / "file_types_per_trip.csv", index=False)

    # anomaly list
    an = []
    for r in df.itertuples():
        if r.max_gap_s > 300:
            an.append((r.partition, r.trip, "large time gap", f"max gap {r.max_gap_s:,.0f} s ({r.max_gap_s/3600:.1f} h)"))
        if r.n_speed_gt250kmh > 0:
            an.append((r.partition, r.trip, "implausible speed spike (GPS jump)",
                       f"{r.n_speed_gt250kmh} steps above 250 km/h (max {r.max_speed_kmh:,.0f} km/h, largest step {r.max_jump_m:,.0f} m)"))
        if r.label_dist_median_m > 30:
            an.append((r.partition, r.trip, "route does not follow GPS", f"median GPS-to-route distance {r.label_dist_median_m:.0f} m"))
        if r.n_gps_outside_network_bbox > 0:
            an.append((r.partition, r.trip, "GPS outside network area", f"{r.n_gps_outside_network_bbox} points"))
        if r.n_repeated_position > 0.2 * r.n_obs:
            an.append((r.partition, r.trip, "many stationary points", f"{r.n_repeated_position} repeated positions"))
    pd.DataFrame(an, columns=["partition", "trip", "anomaly", "detail"]).to_csv(OUT / "anomalies.csv", index=False)

    g = df.groupby("partition")
    summary = {
        "trips": g.size().to_dict(),
        "obs_total": g.n_obs.sum().to_dict(), "obs_min": g.n_obs.min().to_dict(),
        "obs_median": g.n_obs.median().to_dict(), "obs_max": g.n_obs.max().to_dict(),
        "all_files_present": bool(df[[f"has_{e}" for e in EXT]].all().all()),
        "total_mb": {p: float(sum(gg[f"kb_{e}"].sum() for e in EXT) / 1024) for p, gg in g},
        "columns_consistent": {e: sorted(df[f"cols_{e}"].unique().tolist()) for e in EXT},
        "nan": int(df.n_nan.sum()), "out_of_range": int(df.n_out_of_range.sum()), "zero_coords": int(df.n_zero_coords.sum()),
        "duplicate_rows": int(df.n_duplicate_rows.sum()), "repeated_positions": int(df.n_repeated_position.sum()),
        "gps_outside_network_bbox": int(df.n_gps_outside_network_bbox.sum()),
        "non_increasing_t": int(df.n_non_increasing.sum()), "t_integer_all": bool(df.t_integer.all()),
        "t_start_values": sorted(df.t_start.unique().tolist())[:5],
        "steps_total": int((df.n_obs - 1).sum()),
        "frac_dt_1s_overall": float(((df.frac_dt_1s * (df.n_obs - 1)).sum()) / (df.n_obs - 1).sum()),
        "gaps_gt1s": int(df.n_gaps_gt1s.sum()), "gaps_gt5s": int(df.n_gaps_gt5s.sum()),
        "gaps_gt30s": int(df.n_gaps_gt30s.sum()), "gaps_gt300s": int(df.n_gaps_gt300s.sum()),
        "trips_with_gap_gt300s": int((df.max_gap_s > 300).sum()),
        "nodes_min": int(df.n_nodes.min()), "nodes_median": float(df.n_nodes.median()), "nodes_max": int(df.n_nodes.max()),
        "arcs_min": int(df.n_arcs.min()), "arcs_median": float(df.n_arcs.median()), "arcs_max": int(df.n_arcs.max()),
        "arcs_bad_node_id": int(df.n_arcs_bad_node_id.sum()),
        "isolated_nodes_total": int(df.n_isolated_nodes.sum()), "trips_with_isolated": int((df.n_isolated_nodes > 0).sum()),
        "weak_components_median": float(df.n_weak_components.median()), "weak_components_max": int(df.n_weak_components.max()),
        "largest_weak_frac_min": float(df.largest_weak_frac.min()), "largest_weak_frac_median": float(df.largest_weak_frac.median()),
        "largest_strong_frac_min": float(df.largest_strong_frac.min()), "largest_strong_frac_median": float(df.largest_strong_frac.median()),
        "self_loops": int(df.n_self_loops.sum()), "duplicate_arcs": int(df.n_duplicate_arcs.sum()),
        "duplicate_nodes": int(df.n_duplicate_nodes.sum()),
        "twoway_median": float(df.twoway_frac.median()), "seg_len_median_m": float(df.seg_len_median_m.median()),
        "route_all_ids_valid": bool(df.route_ids_valid.all()),
        "route_connected_frac_min": float(df.route_connected_frac.min()),
        "route_connected_frac_mean": float(df.route_connected_frac.mean()),
        "route_len_min": int(df.route_len.min()), "route_len_median": float(df.route_len.median()), "route_len_max": int(df.route_len.max()),
        "routes_with_repeated_arcs": int((df.route_repeated_arcs > 0).sum()),
        "routes_in_largest_strong": int(df.route_in_largest_strong.sum()),
        "obs_per_route_arc_median": float(df.obs_per_route_arc.median()),
        "extent_km_median": float(np.median(np.maximum(df.extent_ew_km, df.extent_ns_km))),
        "extent_km_max": float(np.max(np.maximum(df.extent_ew_km, df.extent_ns_km))),
        "path_km_median": float(df.path_len_km.median()), "path_km_max": float(df.path_len_km.max()),
        "lon_range": [float(df.lon_min.min()), float(df.lon_max.max())],
        "lat_range": [float(df.lat_min.min()), float(df.lat_max.max())],
        "n_anomalies": len(an),
        "steps_gt250kmh": int(df.n_speed_gt250kmh.sum()), "trips_with_gt250kmh": int((df.n_speed_gt250kmh > 0).sum()),
        "median_speed_kmh_range": [float(df.median_speed_kmh.min()), float(df.median_speed_kmh.max())],
    }
    json.dump(summary, open(OUT / "checklist_summary.json", "w"), indent=2)

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(13, 4))
    d2 = df.sort_values(["partition", "trip"])
    cols = d2.partition.map({"train": "C0", "student_test": "C1"})
    ax.bar(range(len(d2)), d2.n_obs, color=cols)
    ax.set_yscale("log"); ax.set_xticks([])
    ax.set(title="GPS observations per trip (blue = train, orange = student-test)", ylabel="observations (log)",
           xlabel="trips")
    fig.tight_layout(); fig.savefig(OUT / "obs_per_trip.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(range(len(d2)), d2.max_gap_s, color=cols); ax[0].set_yscale("log"); ax[0].set_xticks([])
    ax[0].axhline(300, ls="--", c="r", label="5 min"); ax[0].legend()
    ax[0].set(title="Largest time gap per trip", ylabel="seconds (log)", xlabel="trips")
    ax[1].bar(["1 s", "2-5 s", "6-30 s", "31-300 s", ">300 s"],
              [summary["steps_total"] - summary["gaps_gt1s"], summary["gaps_gt1s"] - summary["gaps_gt5s"],
               summary["gaps_gt5s"] - summary["gaps_gt30s"], summary["gaps_gt30s"] - summary["gaps_gt300s"],
               summary["gaps_gt300s"]])
    ax[1].set_yscale("log"); ax[1].set(title="Time steps by interval (all 90 trips)", ylabel="count (log)")
    fig.tight_layout(); fig.savefig(OUT / "time_gaps.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(df.n_nodes, df.n_arcs, c=df.partition.map({"train": "C0", "student_test": "C1"}), s=15)
    ax[0].set(xscale="log", yscale="log", title="Road network size per trip", xlabel="nodes", ylabel="road segments (arcs)")
    ax[1].hist(df.n_weak_components, bins=30); ax[1].set(title="Disconnected components per network", xlabel="weakly connected components", ylabel="trips")
    ax[2].hist(df.largest_strong_frac, bins=20); ax[2].set(title="Largest strongly-connected part", xlabel="fraction of nodes", ylabel="trips")
    fig.tight_layout(); fig.savefig(OUT / "network_checks.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ext = np.maximum(df.extent_ew_km, df.extent_ns_km)
    ax[0].hist(ext, bins=30); ax[0].set(title="Geographical extent per trip (larger side of bounding box)", xlabel="km", ylabel="trips")
    for p, gg in df.groupby("partition"):
        ax[1].scatter((gg.lon_min + gg.lon_max) / 2, (gg.lat_min + gg.lat_max) / 2, label=p, s=18)
    ax[1].set(title="Trip locations (centre)", xlabel="longitude", ylabel="latitude"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "extent_and_locations.png", dpi=120); plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
