"""PathSnap evaluator UI (Laptop 2).

Lists the evaluation trips served by the API, sends the selected trip id to
POST /predict and shows the returned predicted route + server-side metrics.
The UI never reads any data files and never receives reference routes.

Run (Windows CMD, from the repository root):
  set PATHSNAP_API_URL=http://<LAPTOP-1-IP>:8000
  streamlit run ui/app.py
"""
import os

import folium
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap Evaluator", layout="wide")
st.title("PathSnap - GPS map-matching & route reconstruction")

api = st.sidebar.text_input("API URL (Laptop 1)", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"))
api = api.rstrip("/")

try:
    health = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"Connected - model: {health['model']}")
    st.sidebar.caption(f"Partition served: {health['eval_dir']} ({health['n_trips']} trips)")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Cannot reach API at {api}: {e}")
    st.stop()

trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
if not trips:
    st.warning("The API has no evaluation trips in its folder.")
    st.stop()

st.subheader(f"Evaluation trips ({len(trips)})")
trip_id = st.radio("Select a trip", trips, horizontal=True)

if st.button("Run map-matching", type="primary"):
    with st.spinner(f"Sending trip {trip_id} to the API..."):
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["result"] = r.json()

res = st.session_state.get("result")
if res:
    m = res.get("metrics") or {}
    st.subheader(f"Results for trip {res['trip_id']}  ({res['n_observations']} GPS observations)")
    if m:
        c = st.columns(5)
        c[0].metric("Top-1 accuracy (HMM)", f"{m['hmm_top1_accuracy']:.2%}")
        c[1].metric("Top-1 accuracy (classifier only)", f"{m['classifier_top1_accuracy']:.2%}")
        c[2].metric("Route continuity (HMM)", f"{m['hmm_route_continuity']:.2%}")
        c[3].metric("Route F1 (length-weighted)", f"{m['route_f1']:.2%}")
        c[4].metric("Candidate miss rate", f"{m['candidate_miss_rate']:.2%}")
        with st.expander("All metrics returned by the API"):
            st.dataframe(pd.Series(m, name="value").to_frame())
    else:
        st.info("No reference route on the server for this trip - prediction only.")
    st.caption(f"API timings (s): {res['timings_s']} | HMM segments: {res['n_hmm_segments']}, "
               f"breaks: {res['n_hmm_breaks']}")

    gps = res["gps"]
    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    fmap = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=14, control_scale=True)
    fg_gps = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(gps) // 4000)
    for i in range(0, len(gps), step):
        folium.CircleMarker(gps[i], radius=2, color="#1f77b4", fill=True, weight=1,
                            tooltip=f"obs {i}").add_to(fg_gps)
    fg_clf = folium.FeatureGroup(name="Classifier-only prediction", show=False)
    for line in res["classifier_only_route_lines"]:
        folium.PolyLine(line, color="#ff7f0e", weight=3, dash_array="6").add_to(fg_clf)
    fg_pred = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)")
    for line in res["predicted_route_lines"]:
        folium.PolyLine(line, color="#d62728", weight=4, opacity=0.9).add_to(fg_pred)
    for fg in (fg_gps, fg_clf, fg_pred):      # added last = drawn on top: route over the GPS dots
        fg.add_to(fmap)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px;'
        'border:1px solid #888;border-radius:6px;font:12px sans-serif"><b>Legend</b>'
        '<div><span style="color:#1f77b4">&#9679;</span> GPS observation</div>'
        '<div><span style="color:#d62728">&#9473;&#9473;</span> Predicted route (HMM)</div>'
        '<div><span style="color:#ff7f0e">- -</span> Classifier-only</div></div>'))
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    components.html(fmap.get_root().render(), height=620)

    df = pd.DataFrame({"observation": range(len(gps)), "lat": lats, "lon": lons,
                       "predicted_arc_id": res["predicted_arc_per_observation"]})
    st.download_button("Download per-observation predictions (CSV)", df.to_csv(index=False),
                       file_name=f"pathsnap_{res['trip_id']}_predictions.csv")
