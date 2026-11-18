"""Road network utilities: segment geometry, spatial index and shortest paths."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import config, geo


class RoadNetwork:
    """Directed road graph of one trip, in the trip's local metric frame.

    Only arcs whose end nodes lie within GRAPH_MARGIN_M of the trajectory's
    bounding box are indexed (the node files cover a much larger area than the
    trajectory). Arc ids are always the ORIGINAL row indices of the .arcs file.
    """

    def __init__(self, trip, margin_m=config.GRAPH_MARGIN_M, step_m=config.DENSIFY_STEP_M):
        self.trip = trip
        nxy = trip.node_xy
        arcs = trip.arcs
        lo = trip.xy.min(0) - margin_m
        hi = trip.xy.max(0) + margin_m
        inside = np.all((nxy >= lo) & (nxy <= hi), axis=1)
        valid = (arcs[:, 0] >= 0) & (arcs[:, 1] >= 0) & (arcs[:, 0] < len(nxy)) & (arcs[:, 1] < len(nxy))
        keep = valid.copy()
        keep[valid] = inside[arcs[valid, 0]] & inside[arcs[valid, 1]]
        self.arc_ids = np.nonzero(keep)[0]                 # indexed arcs (original ids)
        self.n_nodes = len(nxy)
        self.A = nxy[arcs[:, 0].clip(0, len(nxy) - 1)]      # start point of every arc
        self.B = nxy[arcs[:, 1].clip(0, len(nxy) - 1)]      # end point of every arc
        self.length = np.linalg.norm(self.B - self.A, axis=1)
        d = self.B - self.A
        self.bearing = geo.bearing_xy(d[:, 0], d[:, 1])
        self.out_deg = np.bincount(arcs[keep, 0], minlength=self.n_nodes)
        self.in_deg = np.bincount(arcs[keep, 1], minlength=self.n_nodes)
        # does the opposite-direction arc exist? (two-way road indicator)
        key = arcs[keep, 0].astype(np.int64) * self.n_nodes + arcs[keep, 1]
        rkey = arcs[keep, 1].astype(np.int64) * self.n_nodes + arcs[keep, 0]
        self.twoway = np.zeros(len(arcs), bool)
        self.twoway[self.arc_ids] = np.isin(rkey, key)
        self._build_index(step_m)
        self._build_graph(keep)

    # ---------------- spatial index ----------------
    def _build_index(self, step_m):
        ids = self.arc_ids
        n_samp = np.maximum(2, np.ceil(self.length[ids] / step_m).astype(int) + 1)
        rep = np.repeat(np.arange(len(ids)), n_samp)
        offs = np.concatenate([np.linspace(0, 1, k) for k in n_samp]) if len(ids) else np.zeros(0)
        pts = self.A[ids][rep] + offs[:, None] * (self.B[ids][rep] - self.A[ids][rep])
        self.sample_arc = ids[rep]
        self.tree = cKDTree(pts) if len(pts) else None
        self.half_step = step_m / 2.0

    def candidates(self, xy, radius=config.CANDIDATE_RADIUS_M, k=config.MAX_CANDIDATES):
        """For each point return the k nearest arcs within `radius` (exact
        point-to-segment distance). Returns list of (arc_ids, dist, t, t_raw)."""
        out = []
        if self.tree is None:
            return [(np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0)) for _ in xy]
        hits = self.tree.query_ball_point(xy, r=radius + self.half_step + 1.0)
        for p, h in zip(xy, hits):
            if not h:
                out.append((np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0)))
                continue
            a = np.unique(self.sample_arc[h])
            d, t, traw = geo.point_segment(p[None, :], self.A[a], self.B[a])
            m = d <= radius
            a, d, t, traw = a[m], d[m], t[m], traw[m]
            o = np.argsort(d, kind="stable")[:k]
            out.append((a[o], d[o], t[o], traw[o]))
        return out

    # ---------------- graph / shortest paths ----------------
    def _build_graph(self, keep):
        arcs = self.trip.arcs
        u = arcs[keep, 0]
        v = arcs[keep, 1]
        w = np.maximum(self.length[keep], 0.01)
        # csr_matrix would SUM duplicate (u,v) entries; keep the shortest parallel arc instead
        key = u.astype(np.int64) * self.n_nodes + v
        order = np.lexsort((w, key))
        first = np.ones(len(order), bool)
        first[1:] = key[order][1:] != key[order][:-1]
        sel = order[first]
        u, v, w = u[sel], v[sel], w[sel]
        self.uv_to_arc = dict(zip(zip(u.tolist(), v.tolist()), self.arc_ids[sel].tolist()))
        # compact node indexing (only nodes used by indexed arcs) -> much smaller Dijkstra outputs
        used = np.unique(np.concatenate([u, v]))
        self.local = np.full(self.n_nodes, -1, dtype=np.int64)
        self.local[used] = np.arange(len(used))
        self.global_node = used
        m = len(used)
        self.G = csr_matrix((w, (self.local[u], self.local[v])), shape=(m, m))
        self._sp_cache = {}

    def distances_from(self, node, limit):
        """Shortest network distances from ORIGINAL node id `node`, up to `limit` metres.
        Returns dict {original_node_id: distance}. Results are cached per (node, limit)."""
        key = (int(node), float(limit))
        hit = self._sp_cache.get(key)
        if hit is not None:
            return hit
        ln = self.local[node] if 0 <= node < self.n_nodes else -1
        if ln < 0:
            res = {int(node): 0.0}
        else:
            d = dijkstra(self.G, directed=True, indices=int(ln), limit=limit)
            idx = np.nonzero(np.isfinite(d))[0]
            res = dict(zip(self.global_node[idx].tolist(), d[idx].tolist()))
        if len(self._sp_cache) > 20000:
            self._sp_cache.clear()
        self._sp_cache[key] = res
        return res

    def shortest_path_nodes(self, src, dst, limit=5000.0):
        """Node sequence (original ids) of the shortest path src->dst, or None."""
        ls, ld = self.local[src], self.local[dst]
        if ls < 0 or ld < 0:
            return None
        if ls == ld:
            return [int(src)]
        d, pred = dijkstra(self.G, directed=True, indices=int(ls), limit=limit, return_predecessors=True)
        if not np.isfinite(d[ld]):
            return None
        path = [int(ld)]
        while path[-1] != ls:
            path.append(int(pred[path[-1]]))
        return [int(self.global_node[p]) for p in reversed(path)]

    def arc_connected(self, a, b):
        """True if arc b starts where arc a ends (or a == b)."""
        arcs = self.trip.arcs
        return (a == b) or (arcs[a, 1] == arcs[b, 0])
