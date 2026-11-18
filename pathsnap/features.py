"""Candidate generation and feature engineering.

Problem framing
---------------
Every GPS observation i gets a candidate set C_i = the K nearest road
segments (arcs) within R metres. Each (observation, candidate) pair becomes
one row; the binary target is 1 if the candidate is the observation's
reference arc. A classifier scores every pair; per observation the scores
are normalised over C_i to give P(arc | observation), which (a) gives the
classifier-only prediction (arg-max) and (b) is the HMM emission.

Features (all computed from GPS + road network only, never from labels)
-----------------------------------------------------------------------
dist            perpendicular distance point -> segment (m)
dist_rank       rank of the candidate by distance inside C_i (0 = nearest)
dist_gap        dist - min(dist in C_i)
dist_ratio      dist / (min dist + 1)
n_cands         |C_i|  (dense road areas are harder)
head_diff       |GPS heading - arc bearing| in [0,180] (directed arcs!)
head_diff_w     same, using a +/-3 obs heading (more robust at low speed)
head_valid      1 if the vehicle moved enough for a reliable heading
speed           GPS speed (m/s, central difference)
arc_len         segment length (m)
t_pos           projected position along the arc in [0,1]
t_outside       metres the projection falls beyond the segment ends
dist_prev/next  distance of the previous / next GPS point to the same arc
dist_win        mean distance of points i-3..i+3 to the arc (temporal context)
out_deg/in_deg  degree of the arc's end/start node (junction complexity)
twoway          1 if the opposite-direction arc exists
dt_prev/dt_next time gap to previous/next observation (s)
"""
import numpy as np
import pandas as pd

from . import config, geo

FEATURES = [
    "dist", "dist_rank", "dist_gap", "dist_ratio", "n_cands",
    "head_diff", "head_diff_w", "head_valid", "speed",
    "arc_len", "t_pos", "t_outside",
    "dist_prev", "dist_next", "dist_win",
    "out_deg", "in_deg", "twoway", "dt_prev", "dt_next",
]


def _headings(xy, t, w):
    n = len(xy)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    d = xy[i1] - xy[i0]
    dist = np.linalg.norm(d, axis=1)
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    return geo.bearing_xy(d[:, 0], d[:, 1]), dist, dist / dt


def build_candidate_table(trip, net, labels=None,
                          radius=config.CANDIDATE_RADIUS_M, k=config.MAX_CANDIDATES):
    """Return a DataFrame with one row per (observation, candidate arc)."""
    xy, t = trip.xy, trip.times
    n = len(xy)
    cands = net.candidates(xy, radius, k)
    head1, mv1, speed = _headings(xy, t, 1)
    head3, mv3, _ = _headings(xy, t, 3)
    dt = np.diff(t)
    dt_prev = np.concatenate([[0.0], dt])
    dt_next = np.concatenate([dt, [0.0]])

    obs = np.concatenate([np.full(len(c[0]), i) for i, c in enumerate(cands)]).astype(np.int64)
    if len(obs) == 0:
        return pd.DataFrame(columns=["obs", "arc"] + FEATURES)
    arc = np.concatenate([c[0] for c in cands]).astype(np.int64)
    dist = np.concatenate([c[1] for c in cands])
    tpos = np.concatenate([c[2] for c in cands])
    traw = np.concatenate([c[3] for c in cands])
    rank = np.concatenate([np.arange(len(c[0])) for c in cands])
    ncand = np.concatenate([np.full(len(c[0]), len(c[0])) for c in cands])
    dmin = np.concatenate([np.full(len(c[0]), c[1][0]) for c in cands if len(c[0])])

    A, B = net.A[arc], net.B[arc]
    L = np.maximum(net.length[arc], 1e-6)
    t_out = np.where(traw < 0, -traw, np.where(traw > 1, traw - 1, 0.0)) * L

    def dist_at(offset):
        j = np.clip(obs + offset, 0, n - 1)
        d, _, _ = geo.point_segment(xy[j], A, B)
        return d

    win = [dist_at(o) for o in range(-3, 4)]
    df = pd.DataFrame({
        "obs": obs, "arc": arc,
        "dist": dist, "dist_rank": rank, "dist_gap": dist - dmin,
        "dist_ratio": dist / (dmin + 1.0), "n_cands": ncand,
        "head_diff": geo.angle_diff(head1[obs], net.bearing[arc]),
        "head_diff_w": geo.angle_diff(head3[obs], net.bearing[arc]),
        "head_valid": (mv1[obs] > 2.0).astype(np.int8),
        "speed": speed[obs],
        "arc_len": net.length[arc], "t_pos": tpos, "t_outside": t_out,
        "dist_prev": win[2], "dist_next": win[4], "dist_win": np.mean(win, axis=0),
        "out_deg": net.out_deg[net.trip.arcs[arc, 1]], "in_deg": net.in_deg[net.trip.arcs[arc, 0]],
        "twoway": net.twoway[arc].astype(np.int8),
        "dt_prev": dt_prev[obs], "dt_next": dt_next[obs],
    })
    # heading is meaningless when stationary: replace with a neutral 90 deg
    stat = df["head_valid"].values == 0
    df.loc[stat, "head_diff"] = 90.0
    df.loc[mv3[obs] <= 2.0, "head_diff_w"] = 90.0
    if labels is not None:
        df["y"] = (arc == labels[obs]).astype(np.int8)
    df.insert(0, "trip", trip.trip_id)
    return df
