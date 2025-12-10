
from prefect import flow, task, get_run_logger
import subprocess, os, json, glob, datetime as dt, pathlib

ART_DIR = "orchestration_artifacts"
pathlib.Path(ART_DIR).mkdir(exist_ok=True)

NOTEBOOKS_IN_ORDER = [
    "Task_2_patched.ipynb",
    "Task_3_patched.ipynb",
    "DMML_4_5_6_patched.ipynb",
    "Script-7to9_patched.ipynb",
]

def run_cmd(cmd):
    p = subprocess.run(cmd, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr

@task
def run_notebook(nb_path: str):
    logger = get_run_logger()
    out_nb = os.path.join(ART_DIR, nb_path.replace(".ipynb", "_executed.ipynb"))
    logger.info(f"▶️ Starting {nb_path}")
    code, out, err = run_cmd(["papermill", "-k", "python3", nb_path, out_nb])
    if out: logger.info(out)
    if code != 0:
        if err: logger.error(err)
        raise RuntimeError(f"Failed executing notebook: {nb_path}")
    logger.info(f"✅ Finished {nb_path} -> {out_nb}")
    return out_nb

@task
def write_run_summary():
    summary = {
        "run_id": dt.datetime.now().isoformat(),
        "executed_notebooks": sorted(glob.glob(os.path.join(ART_DIR, "*_executed.ipynb"))),
        "raw": sorted(glob.glob("data/raw/**/*", recursive=True))[:20],
        "validated": sorted(glob.glob("data/validated/*")),
        "transformed": sorted(glob.glob("data/transformed/*")),
        "feature_store": sorted(glob.glob("feature_store/feature_data/*"))[-10:],
        "models": sorted(glob.glob("models/*.pkl")),
        "reports": sorted(glob.glob("reports/*.json")),
    }
    out = os.path.join(ART_DIR, "orchestration_run_summary.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    return out

@flow(name="pipeline-1-to-9-orchestration")
def run_pipeline():
    for nb in NOTEBOOKS_IN_ORDER:
        run_notebook(nb)
    summary_path = write_run_summary()
    get_run_logger().info(f"Run summary written to: {summary_path}")

if __name__ == "__main__":
    run_pipeline()
