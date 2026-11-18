"""Geometry helpers.

Longitude/latitude are angles, so Euclidean distance in degrees is not a
physical distance. Every trip is projected onto a local tangent plane
(equirectangular projection centred on the trip's mean position):

    x = R * (lon - lon0) * cos(lat0)      y = R * (lat - lat0)     [radians]

For trips spanning a few tens of km this keeps local distances (tens of
metres, the scale map-matching works at) accurate to well below 1 %, needs no
external projection library and works anywhere on the globe (the dataset
contains trips on different continents, so one fixed UTM zone would not fit).
"""
import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius in metres


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0 = lon0
        self.lat0 = lat0
        self.kx = EARTH_R * np.cos(np.radians(lat0)) * np.pi / 180.0
        self.ky = EARTH_R * np.pi / 180.0

    def forward(self, lon, lat):
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        return np.column_stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky])

    def inverse(self, xy):
        xy = np.asarray(xy, float)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment(p, a, b):
    """Vectorised point-to-segment projection.

    p, a, b: (..., 2) arrays (broadcastable).
    Returns (distance, t_clipped, t_raw) where t is the position along a->b.
    """
    ab = b - a
    ap = p - a
    L2 = np.sum(ab * ab, axis=-1)
    L2 = np.where(L2 < 1e-12, 1e-12, L2)
    t_raw = np.sum(ap * ab, axis=-1) / L2
    t = np.clip(t_raw, 0.0, 1.0)
    proj = a + t[..., None] * ab
    d = np.linalg.norm(p - proj, axis=-1)
    return d, t, t_raw


def bearing_xy(dx, dy):
    """Compass-style bearing in degrees [0,360) of a vector in the local plane."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Absolute smallest difference between two bearings, in [0,180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
