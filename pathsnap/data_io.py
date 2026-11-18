"""Reading raw PathSnap trip folders.

Each trip folder <id>/ contains:
  <id>.track : lon lat t      (whitespace separated, no header)
  <id>.nodes : lon lat        (row index = node_id)
  <id>.arcs  : from to        (row index = arc_id, directed)
  <id>.route : arc_id         (chronological reference route; optional)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import geo


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N,3) lon, lat, t
    nodes: np.ndarray            # (V,2) lon, lat
    arcs: np.ndarray             # (E,2) from_node, to_node
    route: Optional[np.ndarray]  # (M,) arc ids or None
    proj: "geo.LocalProjection" = field(default=None)
    # projected (metres) versions, filled by .project()
    xy: np.ndarray = field(default=None)        # (N,2)
    node_xy: np.ndarray = field(default=None)   # (V,2)

    def project(self):
        lon0 = float(np.mean(self.track[:, 0]))
        lat0 = float(np.mean(self.track[:, 1]))
        self.proj = geo.LocalProjection(lon0, lat0)
        self.xy = self.proj.forward(self.track[:, 0], self.track[:, 1])
        self.node_xy = self.proj.forward(self.nodes[:, 0], self.nodes[:, 1])
        return self

    @property
    def times(self):
        return self.track[:, 2]


def list_trips(partition_dir):
    partition_dir = Path(partition_dir)
    if not partition_dir.exists():
        return []
    return sorted(p.name for p in partition_dir.iterdir()
                  if p.is_dir() and (p / f"{p.name}.track").exists())


def _load(path, dtype=float, ncols=None):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2 if ncols else 1)
    return arr


def load_trip(partition_dir, trip_id, with_route=True, project=True) -> Trip:
    d = Path(partition_dir) / trip_id
    track = np.loadtxt(d / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    route = None
    rp = d / f"{trip_id}.route"
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=np.int64, ndmin=1)
    trip = Trip(trip_id, track, nodes, arcs, route)
    trip = clean_track(trip)
    if project:
        trip.project()
    return trip


def clean_track(trip: Trip) -> Trip:
    """Minimal, label-free cleaning applied identically to every partition:
    drop rows with non-finite values / implausible coordinates, sort by time,
    and drop exact duplicate timestamps (keep first)."""
    tr = trip.track
    ok = np.all(np.isfinite(tr), axis=1)
    ok &= (np.abs(tr[:, 0]) <= 180) & (np.abs(tr[:, 1]) <= 90)
    tr = tr[ok]
    order = np.argsort(tr[:, 2], kind="stable")
    tr = tr[order]
    keep = np.ones(len(tr), bool)
    keep[1:] = np.diff(tr[:, 2]) > 0
    trip.track = tr[keep]
    return trip
