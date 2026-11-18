"""Folium map generation (used by the EDA/report scripts and by the API)."""
import folium
import numpy as np
from scipy.spatial import cKDTree

COLORS = {"gps": "#1f77b4", "network": "#9e9e9e", "reference": "#2ca02c",
          "predicted": "#d62728", "classifier": "#ff7f0e"}


def _arcs_to_lines(trip, arc_ids):
    """Merge a chronological arc sequence into polylines (break where not connected)."""
    lines, cur = [], []
    prev_to = None
    for a in arc_ids:
        u, v = trip.arcs[a]
        if prev_to is not None and u != prev_to and (not cur or a != cur[-1]):
            if len(cur) > 1:
                lines.append(cur)
            cur = []
        if not cur:
            cur.append(u)
        cur.append(v)
        prev_to = v
    if len(cur) > 1:
        lines.append(cur)
    return [[(trip.nodes[n, 1], trip.nodes[n, 0]) for n in line] for line in lines]


def dedupe_consecutive(seq):
    seq = list(seq)
    return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]


def _legend(items):
    rows = "".join(
        f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>' for name, c in items)
    return folium.Element(
        '<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #888;border-radius:6px;font:12px sans-serif">'
        f'<b>Legend</b>{rows}</div>')


def trip_map(trip, show_network=True, reference_arcs=None, predicted_arcs=None,
             classifier_arcs=None, title=None, network_radius_m=120, max_gps_points=4000):
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="OpenStreetMap",
                   control_scale=True)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;'
            f'padding:4px 8px;border:1px solid #888;border-radius:4px;font:13px sans-serif"><b>{title}</b></div>'))
    legend = []
    if show_network:
        fg = folium.FeatureGroup(name="Road network (near trajectory)", show=True)
        tree = cKDTree(trip.xy)
        mid = (trip.node_xy[trip.arcs[:, 0]] + trip.node_xy[trip.arcs[:, 1]]) / 2
        near = np.nonzero(tree.query(mid, distance_upper_bound=network_radius_m)[0] < np.inf)[0]
        for a in near:
            u, v = trip.arcs[a]
            folium.PolyLine([(trip.nodes[u, 1], trip.nodes[u, 0]), (trip.nodes[v, 1], trip.nodes[v, 0])],
                            color=COLORS["network"], weight=2, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COLORS["network"]))
    if reference_arcs is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        for line in _arcs_to_lines(trip, dedupe_consecutive(reference_arcs)):
            folium.PolyLine(line, color=COLORS["reference"], weight=7, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))
    # layers added later are drawn on top: network, reference, GPS, classifier-only, predicted
    fg = folium.FeatureGroup(name="Raw GPS observations", show=True)
    step = max(1, int(np.ceil(len(lat) / max_gps_points)))
    for i in range(0, len(lat), step):
        folium.CircleMarker((lat[i], lon[i]), radius=2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.8, weight=1, tooltip=f"obs {i}, t={trip.track[i, 2]:.0f}s").add_to(fg)
    fg.add_to(m)
    legend.append(("GPS observation", COLORS["gps"]))
    if classifier_arcs is not None:
        fg = folium.FeatureGroup(name="Classifier-only prediction", show=False)
        for line in _arcs_to_lines(trip, dedupe_consecutive(classifier_arcs)):
            folium.PolyLine(line, color=COLORS["classifier"], weight=3, opacity=0.9, dash_array="6").add_to(fg)
        fg.add_to(m)
        legend.append(("Classifier-only prediction", COLORS["classifier"]))
    if predicted_arcs is not None:
        fg = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)", show=True)
        for line in _arcs_to_lines(trip, dedupe_consecutive(predicted_arcs)):
            folium.PolyLine(line, color=COLORS["predicted"], weight=3.5, opacity=0.9).add_to(fg)
        fg.add_to(m)
        legend.append(("Predicted route", COLORS["predicted"]))
    folium.Marker((lat[0], lon[0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    return m
