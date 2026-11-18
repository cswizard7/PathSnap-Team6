"""Deriving observation-level reference labels from the .route file.

The .route file is a chronological list of arc ids, NOT one label per GPS row
(a trip with 1 315 GPS rows has a 525-arc route). To obtain a reference road
segment for every observation we solve a monotone alignment problem:

    choose j_0 <= j_1 <= ... <= j_{N-1}   (indices into the route)
    minimising  sum_i dist(p_i, segment(route[j_i]))

i.e. every GPS point is assigned to a route arc, and the assignment may only
move forward along the route (the vehicle never travels backwards in time).
Solved exactly by dynamic programming with a running prefix-minimum,
O(N*M) time.

These labels are used ONLY for training and for server-side scoring; they
never enter the prediction pipeline.
"""
import numpy as np

from . import geo


def align_route(trip, chunk=400):
    route = trip.route
    if route is None or len(route) == 0:
        return None, None
    nxy = trip.node_xy
    A = nxy[trip.arcs[route, 0]]
    B = nxy[trip.arcs[route, 1]]
    N, M = len(trip.xy), len(route)
    back = np.empty((N, M), dtype=np.int32)
    prev = None
    for s in range(0, N, chunk):
        P = trip.xy[s:s + chunk]
        C, _, _ = geo.point_segment(P[:, None, :], A[None, :, :], B[None, :, :])
        C = C.astype(np.float64)
        for r in range(len(P)):
            i = s + r
            if prev is None:
                cur = C[r].copy()
                back[i] = np.arange(M)
            else:
                # prefix minimum of previous row and its argmin
                idx = np.arange(M)
                is_new_min = np.empty(M, bool)
                pm = np.minimum.accumulate(prev)
                is_new_min[0] = True
                is_new_min[1:] = prev[1:] <= pm[:-1]
                arg = np.maximum.accumulate(np.where(is_new_min, idx, 0))
                back[i] = arg
                cur = C[r] + pm
            prev = cur
    j = int(np.argmin(prev))
    path = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    labels = route[path]
    dist, _, _ = geo.point_segment(trip.xy, nxy[trip.arcs[labels, 0]], nxy[trip.arcs[labels, 1]])
    return labels, dist
