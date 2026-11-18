"""Step 6 - Folium maps of the final pipeline on student-testing trips.

Each map has layers (toggle with the layer control): raw GPS observations,
road network near the trajectory, reference route, classifier-only
prediction and the HMM/Viterbi predicted route, plus a legend.
Reference routes are drawn here for the REPORT only; the API/UI never send them.

Run:  python scripts/06_make_maps.py              (all 10 student-test trips)
      python scripts/06_make_maps.py 00000000     (one trip)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pathsnap import config, data_io, visual
from pathsnap.pipeline import PathSnapModel, score_trip

OUT = config.RESULTS_DIR / "maps"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    model = PathSnapModel()
    trips = sys.argv[1:] or data_io.list_trips(config.STUDENT_TEST_DIR)
    rows = []
    for tid in trips:
        pred = model.predict_trip(config.STUDENT_TEST_DIR, tid)
        sc = score_trip(config.STUDENT_TEST_DIR, tid, pred)
        ref = data_io.load_trip(config.STUDENT_TEST_DIR, tid)
        trip = pred["trip"]
        m = visual.trip_map(trip, show_network=True, reference_arcs=ref.route,
                            predicted_arcs=pred["route"], classifier_arcs=pred["classifier_arcs"],
                            title=f"Student-test trip {tid}: HMM top-1 {sc['hmm_top1_accuracy']:.1%}, "
                                  f"classifier-only {sc['classifier_top1_accuracy']:.1%}")
        m.save(str(OUT / f"student_test_{tid}.html"))
        rows.append({"trip": tid, **sc, **pred["timings"]})
        print(tid, {k: round(v, 4) for k, v in sc.items()})
    pd.DataFrame(rows).to_csv(OUT / "maps_trip_scores.csv", index=False)


if __name__ == "__main__":
    main()
