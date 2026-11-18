"""HMM + Viterbi route reconstruction.

Hidden states
    At observation i the hidden state is "the vehicle is on candidate arc c,
    at the projected position t_c of the GPS point", for c in the candidate set
    C_i. The state index is the candidate's row in the observation's candidate
    list, so the candidate -> road-segment correspondence is preserved.

Emission  log P(z_i | c)
    w_e * log( p_c / sum_{c' in C_i} p_c' )
    where p_c is the classifier probability for the (observation, candidate)
    row. The classifier already combines distance, heading, context, etc.

Transition  log P(c_i | c_{i-1})
    Newson & Krumm (2009): the network distance travelled between the two
    projected positions should be close to the straight-line distance between
    the two GPS points:
        log P = -|d_route - d_gc| / beta
    d_route = remaining length on arc a + shortest path(end(a) -> start(b))
              + distance along arc b; on the same arc it is the difference of
              positions (small backward moves <= BACK_SLACK_M are tolerated as
              GPS noise). Unreachable pairs (no path within the search limit)
              get log P = -inf.

Gaps / breaks
    * observations with no candidate get no state (predicted arc = -1) and the
      chain bridges over them;
    * if dt > GAP_SECONDS the chain is split and each part decoded separately;
    * if no transition from step i-1 to step i is feasible (disconnected
      segments, e.g. GPS jump or missing road), the chain is broken and
      restarted at i from emissions only. Breaks are counted and reported.

Decoding
    Standard Viterbi in log-space:
        V_i(b) = E_i(b) + max_a [ V_{i-1}(a) + T(a,b) ],  back-pointer = argmax
    followed by back-tracking from argmax_b V_last(b).
"""
from dataclasses import dataclass

import numpy as np

GAP_SECONDS = 60.0
BACK_SLACK_M = 15.0


@dataclass
class HMMParams:
    beta: float = 10.0          # transition scale (m)
    emission_weight: float = 1.0
    min_prob: float = 1e-6      # floor for normalised candidate probabilities
    max_route_factor: float = 4.0   # search limit = max(300, factor * d_gc + 100) m


def _group_candidates(cand_df, proba, n_obs, params):
    """Per observation: arrays of arcs, positions t and log-emissions."""
    obs = cand_df["obs"].values
    order = np.argsort(obs, kind="stable")
    obs_s = obs[order]
    arcs = cand_df["arc"].values[order]
    tpos = cand_df["t_pos"].values[order]
    p = np.maximum(np.asarray(proba)[order], params.min_prob)
    bounds = np.searchsorted(obs_s, np.arange(n_obs + 1))
    out = []
    for i in range(n_obs):
        s, e = bounds[i], bounds[i + 1]
        if e <= s:
            out.append(None)
            continue
        pi = p[s:e] / p[s:e].sum()
        out.append((arcs[s:e], tpos[s:e], params.emission_weight * np.log(np.maximum(pi, params.min_prob))))
    return out


def _transition_matrix(net, prev, cur, d_gc, params):
    a_arcs, a_t, _ = prev
    b_arcs, b_t, _ = cur
    La = net.length[a_arcs]
    Lb = net.length[b_arcs]
    limit = max(300.0, params.max_route_factor * d_gc + 100.0)
    limit = float(np.ceil(limit / 100.0) * 100.0)        # quantise -> better cache reuse
    arcs_tab = net.trip.arcs
    T = np.full((len(a_arcs), len(b_arcs)), -np.inf)
    for ia, a in enumerate(a_arcs):
        rem = (1.0 - a_t[ia]) * La[ia]
        dist_from_end = None
        for ib, b in enumerate(b_arcs):
            if a == b:
                fwd = (b_t[ib] - a_t[ia]) * La[ia]
                if fwd >= -BACK_SLACK_M:
                    d = abs(fwd)
                    T[ia, ib] = -abs(d - d_gc) / params.beta
                    continue
            if dist_from_end is None:
                dist_from_end = net.distances_from(arcs_tab[a, 1], limit)
            sp = dist_from_end.get(int(arcs_tab[b, 0]))
            if sp is None:
                continue
            d = rem + sp + b_t[ib] * Lb[ib]
            T[ia, ib] = -abs(d - d_gc) / params.beta
    return T


def viterbi(trip, net, cand_df, proba, params: HMMParams = HMMParams()):
    """Decode the most likely arc for every observation.

    Returns dict with 'arcs' (per-observation arc id, -1 if none),
    'n_breaks', 'n_segments'."""
    n = len(trip.xy)
    groups = _group_candidates(cand_df, proba, n, params)
    xy, t = trip.xy, trip.times
    result = np.full(n, -1, dtype=np.int64)
    n_breaks = 0
    n_segments = 0

    seg_obs, seg_back = [], []   # observations of current chain and back-pointers
    V = None
    last_i = None

    def flush():
        nonlocal seg_obs, seg_back, V
        if not seg_obs:
            return
        k = int(np.argmax(V))
        for j in range(len(seg_obs) - 1, -1, -1):
            i = seg_obs[j]
            result[i] = groups[i][0][k]
            if j > 0:
                k = seg_back[j][k]
        seg_obs, seg_back, V = [], [], None

    for i in range(n):
        g = groups[i]
        if g is None:
            continue
        new_chain = V is None or (t[i] - t[last_i]) > GAP_SECONDS
        if not new_chain:
            d_gc = float(np.linalg.norm(xy[i] - xy[last_i]))
            T = _transition_matrix(net, groups[last_i], g, d_gc, params)
            S = V[:, None] + T
            best = np.max(S, axis=0)
            if not np.any(np.isfinite(best)):
                n_breaks += 1
                new_chain = True
            else:
                bp = np.argmax(S, axis=0)
                V = best + g[2]
                seg_obs.append(i)
                seg_back.append(bp)
        if new_chain:
            flush()
            n_segments += 1
            V = g[2].copy()
            seg_obs.append(i)
            seg_back.append(None)
        last_i = i
    flush()
    return {"arcs": result, "n_breaks": n_breaks, "n_segments": n_segments}


def reconstruct_route(net, obs_arcs):
    """Turn per-observation arcs into a continuous arc sequence by inserting the
    shortest connecting path between consecutive (different, non-adjacent) arcs."""
    seq = [int(a) for a in obs_arcs if a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    arcs_tab = net.trip.arcs
    route = []
    for i, a in enumerate(seq):
        if route:
            prev = route[-1]
            u, v = int(arcs_tab[prev, 1]), int(arcs_tab[a, 0])
            if u != v:
                nodes = net.shortest_path_nodes(u, v, limit=5000.0)
                if nodes is not None:
                    for x, y in zip(nodes[:-1], nodes[1:]):
                        e = net.uv_to_arc.get((x, y))
                        if e is not None:
                            route.append(e)
        route.append(a)
    return route
