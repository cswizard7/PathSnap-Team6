# Evaluation-day runbook (two laptops)

All commands are for **Windows Command Prompt (cmd.exe)**, run from the repository root.

## The day before / morning

1. Both laptops: clone the repo, create the venv and `pip install -r requirements.txt` (section 2 of README).
2. Laptop 1: copy `data\train` and `data\student_test` in place (needed only for re-running experiments; the API itself needs only `models\` + the evaluation folder).
3. Laptop 1: dry run with the student-testing trips:
   ```bat
   set PATHSNAP_EVAL_DIR=data\student_test
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
4. Laptop 2: `set PATHSNAP_API_URL=http://<laptop-1-ip>:8000` then `streamlit run ui\app.py`; run all 10 trips once.
5. Keep a phone hotspot ready in case the venue Wi-Fi blocks laptop-to-laptop traffic (client isolation).

## During the evaluation

1. Copy the 10 evaluator trip folders into `data\evaluator_test\` on **Laptop 1** (one folder per trip, files unchanged).
2. Laptop 1 (CMD window stays visible - evaluators want to see the request log):
   ```bat
   .venv\Scripts\activate
   set PATHSNAP_EVAL_DIR=data\evaluator_test
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   The startup log must list the 10 trip ids.
3. `ipconfig` -> note the IPv4 address.
4. Laptop 2:
   ```bat
   .venv\Scripts\activate
   set PATHSNAP_API_URL=http://<laptop-1-ip>:8000
   streamlit run ui\app.py
   ```
5. Evaluator selects a trip -> **Run map-matching** -> map + metrics appear; the Laptop 1 CMD shows
   `POST /predict from <laptop-2-ip>: trip=... -> running pipeline` and the accuracy line.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI: "Cannot reach API" | Same network? Correct IP? Try `curl http://<ip>:8000/health` on Laptop 2. Allow Python through Windows Firewall (Private) or add the port rule from the README. |
| API startup lists 0 trips | Folder layout must be `data\evaluator_test\<id>\<id>.track` (not nested one level deeper after unzipping). |
| `ModuleNotFoundError` | venv not activated (`.venv\Scripts\activate`) or `pip install -r requirements.txt` not run. |
| Model fails to load (version error) | `pip install scikit-learn==1.9.1`; or rebuild it: `python scripts\02_build_features.py` then `python scripts\07_refit_final_model.py`. |
| Port 8000 busy | `uvicorn api.main:app --host 0.0.0.0 --port 8001` and use `:8001` in the UI. |
| Map tiles blank | Laptop 2 needs internet for OpenStreetMap tiles; the route/GPS layers still draw. |

## Screenshots needed for the report (section 7)

* The UI showing a trip result (map + metric cards).
* The Laptop 1 CMD window showing `GET /trips` and `POST /predict ...` log lines with accuracy.
