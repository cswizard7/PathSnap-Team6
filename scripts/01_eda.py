"""Step 1 - Dataset exploration (training + student-testing partitions only).

Outputs (results/eda/):
  inventory.csv           one row per trip: files, sizes, counts, extents, gaps, anomalies
  network_stats.csv       road-network size and connectivity per trip
  label_quality.csv       GPS-to-reference-route distance per trip (route alignment)
  data_quality_notes.md   automatically generated summary of issues found
  *.png                   plots
  maps/*.html             Folium maps of selected trips (GPS + road network + reference route)

Run:  python scripts/01_eda.py
"""
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

from pathsnap import config, data_io, geo, labels, visual

OUT = config.RESULTS_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "maps").mkdir(exist_ok=True)


def raw_checks(part_dir, tid):
    """Checks on the RAW files (before cleaning)."""
    d = Path(part_dir) / tid
    files = {ext: d / f"{tid}.{ext}" for ext in ["track", "nodes", "arcs", "route"]}
    raw = np.genfromtxt(files["track"], invalid_raise=False)
    n_raw = len(raw)
    n_nan = int(np.sum(~np.isfinite(raw).all(axis=1)))
    t = raw[:, 2]
    dt = np.diff(t)
    return {
        "trip": tid,
        **{f"has_{e}": f.exists() for e, f in files.items()},
        **{f"kb_{e}": round(f.stat().st_size / 1024, 1) if f.exists() else np.nan for e, f in files.items()},
        "n_obs_raw": n_raw, "n_rows_with_nan": n_nan,
        "n_nonmonotonic_t": int(np.sum(dt <= 0)),
        "duration_s": float(t[-1] - t[0]),
        "dt_1s_frac": float(np.mean(dt == 1)), "dt_max_s": float(dt.max()),
        "n_gaps_gt5s": int(np.sum(dt > 5)), "n_gaps_gt30s": int(np.sum(dt > 30)),
        "lon_min": raw[:, 0].min(), "lon_max": raw[:, 0].max(),
        "lat_min": raw[:, 1].min(), "lat_max": raw[:, 1].max(),
    }


def network_checks(trip):
    V = len(trip.nodes)
    a = trip.arcs
    G = csr_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(V, V))
    n_weak, lab_w = connected_components(G, directed=True, connection="weak")
    n_strong, lab_s = connected_components(G, directed=True, connection="strong")
    deg = np.bincount(a[:, 0], minlength=V) + np.bincount(a[:, 1], minlength=V)
    seg_len = np.linalg.norm(trip.node_xy[a[:, 1]] - trip.node_xy[a[:, 0]], axis=1)
    key = set(map(tuple, a.tolist()))
    twoway = np.mean([(v, u) in key for u, v in a.tolist()])
    out = {
        "trip": trip.trip_id, "n_nodes": V, "n_arcs": len(a),
        "n_isolated_nodes": int(np.sum(deg == 0)),
        "n_weak_components": n_weak,
        "largest_weak_comp_frac": float(np.bincount(lab_w).max() / V),
        "n_strong_components": n_strong,
        "largest_strong_comp_frac": float(np.bincount(lab_s).max() / V),
        "twoway_arc_frac": float(twoway),
        "seg_len_median_m": float(np.median(seg_len)), "seg_len_max_m": float(seg_len.max()),
        "n_self_loops": int(np.sum(a[:, 0] == a[:, 1])),
        "n_duplicate_arcs": int(len(a) - len(key)),
    }
    if trip.route is not None:
        r = trip.route
        out["route_len_arcs"] = len(r)
        out["route_unique_arcs"] = len(np.unique(r))
        out["route_connected_frac"] = float(np.mean(a[r[:-1], 1] == a[r[1:], 0])) if len(r) > 1 else 1.0
        out["route_length_km"] = float(seg_len[r].sum() / 1000)
    return out


def main():
    rows, nets, qual = [], [], []
    for part, pdir in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        for tid in data_io.list_trips(pdir):
            r = raw_checks(pdir, tid)
            r["partition"] = part
            trip = data_io.load_trip(pdir, tid)
            r["n_obs_clean"] = len(trip.xy)
            step = np.linalg.norm(np.diff(trip.xy, axis=0), axis=1)
            sp = step / np.maximum(np.diff(trip.times), 1e-6)
            r["path_len_km"] = step.sum() / 1000
            r["speed_median_ms"] = float(np.median(sp))
            r["speed_p99_ms"] = float(np.percentile(sp, 99))
            r["n_jumps_gt50ms"] = int(np.sum(sp > 50))
            r["stationary_frac"] = float(np.mean(step < 0.5))
            ext = trip.xy.max(0) - trip.xy.min(0)
            r["extent_ew_km"], r["extent_ns_km"] = ext / 1000
            rows.append(r)
            n = network_checks(trip)
            n["partition"] = part
            nets.append(n)
            lab, d = labels.align_route(trip)
            qual.append({"trip": tid, "partition": part,
                         "label_dist_median_m": float(np.median(d)),
                         "label_dist_p90_m": float(np.percentile(d, 90)),
                         "label_dist_max_m": float(d.max()),
                         "frac_obs_gt50m": float(np.mean(d > 50))})
            np.save(OUT / f"_labeldist_{part}_{tid}.npy", d.astype(np.float32))
            print(f"{part} {tid}: n={len(trip.xy)} label-dist median={np.median(d):.1f} m")

    inv = pd.DataFrame(rows)
    net = pd.DataFrame(nets)
    lq = pd.DataFrame(qual)
    inv.to_csv(OUT / "inventory.csv", index=False)
    net.to_csv(OUT / "network_stats.csv", index=False)
    lq.to_csv(OUT / "label_quality.csv", index=False)

    # ---------- summary tables ----------
    summ = inv.groupby("partition").agg(
        trips=("trip", "count"), obs_total=("n_obs_clean", "sum"),
        obs_min=("n_obs_clean", "min"), obs_median=("n_obs_clean", "median"), obs_max=("n_obs_clean", "max"),
        nan_rows=("n_rows_with_nan", "sum"), nonmonotonic=("n_nonmonotonic_t", "sum"),
        gaps_gt5s=("n_gaps_gt5s", "sum"), gaps_gt30s=("n_gaps_gt30s", "sum"),
        dt_1s_frac_mean=("dt_1s_frac", "mean"))
    summ.to_csv(OUT / "summary_by_partition.csv")
    print(summ)

    # ---------- plots ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, g in inv.groupby("partition"):
        ax[0].hist(g["n_obs_clean"], bins=30, alpha=.6, label=part)
        ax[1].hist(g["duration_s"] / 60, bins=30, alpha=.6, label=part)
        ax[2].hist(g["speed_median_ms"] * 3.6, bins=30, alpha=.6, label=part)
    ax[0].set(title="GPS observations per trip", xlabel="observations", ylabel="trips")
    ax[1].set(title="Trip duration", xlabel="minutes")
    ax[2].set(title="Median speed per trip", xlabel="km/h")
    for a in ax:
        a.legend()
    fig.tight_layout(); fig.savefig(OUT / "trip_size_distributions.png", dpi=120); plt.close(fig)

    # sampling interval distribution (train)
    dts = []
    for tid in data_io.list_trips(config.TRAIN_DIR):
        t = np.loadtxt(config.TRAIN_DIR / tid / f"{tid}.track")[:, 2]
        dts.append(np.diff(t))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(figsize=(7, 4))
    vals, cnts = np.unique(np.clip(dts, 0, 11), return_counts=True)
    ax.bar([str(int(v)) if v < 11 else ">10" for v in vals], cnts)
    ax.set_yscale("log"); ax.set(title=f"Sampling interval (train, {len(dts):,} steps)", xlabel="seconds", ylabel="count (log)")
    fig.tight_layout(); fig.savefig(OUT / "sampling_intervals_train.png", dpi=120); plt.close(fig)

    # GPS-to-reference distance (train) -> justifies candidate radius
    d_all = np.concatenate([np.load(OUT / f"_labeldist_train_{t}.npy")
                            for t in lq[lq.partition == "train"].trip])
    good = lq[(lq.partition == "train") & (lq.label_dist_median_m <= 30)].trip
    d_good = np.concatenate([np.load(OUT / f"_labeldist_train_{t}.npy") for t in good])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(np.clip(d_good, 0, 80), bins=80)
    for q in [90, 99, 99.9]:
        v = np.percentile(d_good, q)
        ax[0].axvline(v, ls="--", c="k"); ax[0].text(v, ax[0].get_ylim()[1] * .8, f" p{q}={v:.0f}m", fontsize=8)
    ax[0].set(title="GPS point -> aligned reference arc distance (train, clean trips)", xlabel="metres")
    ax[1].bar(lq.trip + "\n" + lq.partition.str[:2], lq.label_dist_median_m)
    ax[1].axhline(30, c="r", ls="--", label="30 m quality threshold")
    ax[1].set(title="Median GPS-to-reference distance per trip", ylabel="metres"); ax[1].legend()
    ax[1].set_xticks([])
    fig.tight_layout(); fig.savefig(OUT / "gps_to_reference_distance.png", dpi=120); plt.close(fig)
    pd.Series({f"p{q}": float(np.percentile(d_good, q)) for q in [50, 75, 90, 95, 99, 99.5, 99.9]}).to_csv(
        OUT / "gps_noise_percentiles_train_clean.csv", header=["metres"])

    # geographic distribution of trips
    fig, ax = plt.subplots(figsize=(8, 5))
    for part, g in inv.groupby("partition"):
        ax.scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, label=part, alpha=.7)
    ax.set(title="Trip centres (lon/lat)", xlabel="longitude", ylabel="latitude"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "trip_locations.png", dpi=120); plt.close(fig)

    # trajectory + network plots for a few trips
    for part, pdir, tid in [("train", config.TRAIN_DIR, "00000001"),
                            ("train", config.TRAIN_DIR, "00000050"),
                            ("student_test", config.STUDENT_TEST_DIR, "00000000")]:
        trip = data_io.load_trip(pdir, tid)
        fig, ax = plt.subplots(figsize=(8, 8))
        lo, hi = trip.track[:, :2].min(0) - 0.003, trip.track[:, :2].max(0) + 0.003
        a = trip.arcs
        m = np.all((trip.nodes[a[:, 0]] >= lo) & (trip.nodes[a[:, 0]] <= hi), axis=1)
        for u, v in a[m]:
            ax.plot(trip.nodes[[u, v], 0], trip.nodes[[u, v], 1], c="#bbbbbb", lw=.5, zorder=1)
        r = trip.route
        for u, v in a[r]:
            ax.plot(trip.nodes[[u, v], 0], trip.nodes[[u, v], 1], c="green", lw=2.5, zorder=2)
        ax.scatter(trip.track[:, 0], trip.track[:, 1], s=3, c="blue", zorder=3, label="GPS")
        ax.plot([], [], c="green", label="reference route"); ax.plot([], [], c="#bbbbbb", label="road network")
        ax.set(title=f"{part} trip {tid}", xlabel="longitude", ylabel="latitude"); ax.legend()
        ax.set_aspect(1 / np.cos(np.radians(trip.track[:, 1].mean())))
        fig.tight_layout(); fig.savefig(OUT / f"trajectory_{part}_{tid}.png", dpi=120); plt.close(fig)
        visual.trip_map(trip, show_network=True, reference_arcs=trip.route,
                        title=f"{part} trip {tid}: raw GPS, road network and reference route"
                        ).save(str(OUT / "maps" / f"eda_{part}_{tid}.html"))

    # network size plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(net.n_nodes, net.n_arcs, c=(net.partition == "train").map({True: "C0", False: "C1"}))
    ax[0].set(xscale="log", yscale="log", title="Road network size per trip", xlabel="nodes", ylabel="arcs")
    ax[1].hist(net.largest_strong_comp_frac, bins=20)
    ax[1].set(title="Largest strongly-connected component (fraction of nodes)", xlabel="fraction")
    fig.tight_layout(); fig.savefig(OUT / "network_size_connectivity.png", dpi=120); plt.close(fig)

    # ---------- automatic data-quality notes ----------
    bad = lq[lq.label_dist_median_m > 30]
    notes = [
        "# Data-quality notes (auto-generated by scripts/01_eda.py)", "",
        f"* Partitions analysed: train ({(inv.partition == 'train').sum()} trips), "
        f"student_test ({(inv.partition == 'student_test').sum()} trips). Evaluator-testing data was not used.",
        f"* Rows with missing/non-numeric values: {int(inv.n_rows_with_nan.sum())}.",
        f"* Non-increasing timestamps: {int(inv.n_nonmonotonic_t.sum())}.",
        f"* Mean fraction of 1-second steps: {inv.dt_1s_frac.mean():.3f}; gaps >5 s: {int(inv.n_gaps_gt5s.sum())}; gaps >30 s: {int(inv.n_gaps_gt30s.sum())}; largest gap {inv.dt_max_s.max():.0f} s.",
        f"* Trips spread over several countries/continents (lon {inv.lon_min.min():.1f}..{inv.lon_max.max():.1f}, lat {inv.lat_min.min():.1f}..{inv.lat_max.max():.1f}) -> per-trip local projection.",
        f"* Every trip has its own road-network files; arc ids are trip-local.",
        f"* Reference routes are connected arc chains (mean consecutive-connected fraction {net.route_connected_frac.mean():.3f}).",
        f"* Two-way roads are stored as two opposite directed arcs (median two-way fraction {net.twoway_arc_frac.median():.2f}): the nearest segment alone cannot tell the direction -> heading features.",
        f"* GPS->reference distance on clean training trips: p50={np.percentile(d_good, 50):.1f} m, p90={np.percentile(d_good, 90):.1f} m, p99={np.percentile(d_good, 99):.1f} m, p99.9={np.percentile(d_good, 99.9):.1f} m.",
        f"* Trips whose reference route does not follow the GPS (median distance > 30 m): "
        + (", ".join(f"{r.trip} ({r.partition}, {r.label_dist_median_m:.0f} m)" for r in bad.itertuples()) or "none"),
        "  These labels are unreliable; such TRAIN trips are excluded from model fitting (student-test trips are kept and reported as-is).",
    ]
    (OUT / "data_quality_notes.md").write_text("\n".join(notes), encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
