"""Rebuild the Colab Optuna study of Task 4 from its console log. Plan: docs/TASK4_PLAN.md B.2.

The original Colab study database is lost. What survives is a table transcribed from the console log
(studies/task4_cgan/trials_from_console_log.csv: trials 6 to 25, each COMPLETE with value + parameters or
PRUNED without anything else). This script writes that table into a NEW SQLite study so the usual Optuna
tools (trials.csv, plots) can be used on it.

Run:  python -m genai.tasks.task4.log_study [--csv ...] [--out-db studies/task4_cgan/task4_cgan.db] [--overwrite]

* Trials 0 to 5 are added as FAIL placeholders with no parameters. They exist so that Optuna's trial numbers
  equal the log's numbers (6 to 25) and "Trials run: 26" matches. This rests on the student's RECOLLECTION
  (the first Colab session crashed before training because of a filename bug); the log itself has no trace of them.
* The ranges used for the parameter distributions are the repository ranges of configs/task4_cgan.yaml;
  the original Colab ranges are unknown.
* The study lives in its OWN file. It is never used as storage by tune.py.
* No value is invented: PRUNED trials have no value and no intermediate values because the log has none.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import optuna
import yaml

from genai.tasks.task1.tune import export_study
from genai.tasks.task4.config import resolve
from genai.tasks.task4.tune import COMPLETE, FAIL, PRUNED, print_study_summary

STUDY_NAME = "task4_cgan"
N_PLACEHOLDERS = 6            # trials 0..5 (FAIL placeholders); the CSV starts at trial 6
PLACEHOLDER_NOTE = ("crashed before training (filename bug) in the first Colab session; "
                    "parameters not recorded")
SOURCE_NOTE = "rebuilt from the Colab console log; original DB lost"
RANGES_NOTE = "repository ranges of B.1; the original Colab ranges are unknown"


def distribution_for(spec: dict) -> optuna.distributions.BaseDistribution:
    """The Optuna distribution of one tuned_params entry (same meaning as suggest_params in tune.py)."""
    if spec["type"] == "float":
        return optuna.distributions.FloatDistribution(spec["low"], spec["high"], log=spec.get("log", False))
    if spec["type"] == "int":
        return optuna.distributions.IntDistribution(spec["low"], spec["high"], log=spec.get("log", False))
    if spec["type"] == "categorical":
        return optuna.distributions.CategoricalDistribution(spec["choices"])
    raise ValueError(f"unknown parameter type {spec['type']!r} for {spec['name']}")


def parse_value(spec: dict, text: str):
    """CSV text -> Python value of the right type (float / int; a categorical keeps the type of its choices)."""
    if spec["type"] == "float":
        return float(text)
    if spec["type"] == "int":
        return int(text)
    return type(spec["choices"][0])(text)            # '8' -> 8 for choices [8, 16, 32]


def read_rows(csv_path: Path) -> list:
    """The CSV rows as dicts, sorted by trial number."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sorted(csv.DictReader(f), key=lambda r: int(r["trial"]))


def build_log_study(csv_path, out_db, config_path, overwrite: bool = False) -> optuna.Study:
    """Create the study `task4_cgan` in the SQLite file out_db from the CSV and return it."""
    csv_path, out_db, config_path = Path(csv_path), Path(out_db), Path(config_path)
    space = yaml.safe_load(config_path.read_text(encoding="utf-8"))["tuned_params"]
    rows = read_rows(csv_path)
    if [int(r["trial"]) for r in rows] != list(range(N_PLACEHOLDERS, N_PLACEHOLDERS + len(rows))):
        raise ValueError(f"{csv_path}: trial numbers must run consecutively from {N_PLACEHOLDERS}")

    if out_db.exists():
        if not overwrite:
            raise FileExistsError(f"{out_db} exists; pass --overwrite to delete it and rebuild "
                                  "(never append: that would duplicate trials)")
        out_db.unlink()
    out_db.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(study_name=STUDY_NAME, storage=f"sqlite:///{out_db.as_posix()}",
                                direction="minimize")
    study.set_user_attr("source", SOURCE_NOTE)
    study.set_user_attr("search_ranges", RANGES_NOTE)

    # Trials 0..5: FAIL placeholders (the student's recollection, see the module docstring).
    for number in range(N_PLACEHOLDERS):
        study.add_trial(optuna.trial.create_trial(
            state=FAIL, user_attrs={"note": PLACEHOLDER_NOTE, "original_trial_number": number,
                                    "source": "placeholder, not in the console log"}))

    # Trials 6..25 from the CSV, in number order. add_trial numbers trials consecutively from 0,
    # so the trial added now gets exactly the number the log gives it (checked below).
    for row in rows:
        attrs = {"original_trial_number": int(row["trial"]), "source": row["source"]}
        if row["state"] == "COMPLETE":
            params = {s["name"]: parse_value(s, row[s["name"]]) for s in space}
            trial = optuna.trial.create_trial(
                state=COMPLETE, value=float(row["value_val_l1"]), params=params,
                distributions={s["name"]: distribution_for(s) for s in space}, user_attrs=attrs)
        elif row["state"] == "PRUNED":
            trial = optuna.trial.create_trial(state=PRUNED, user_attrs=attrs)    # the log has no values for it
        else:
            raise ValueError(f"trial {row['trial']}: unexpected state {row['state']!r}")
        study.add_trial(trial)
        assert study.trials[-1].number == int(row["trial"]), "Optuna trial number differs from the log's"

    # ---- checks before anything is reported ----
    states = [t.state for t in study.trials]
    assert len(states) == 26, f"expected 26 trials, got {len(states)}"
    assert (states.count(COMPLETE), states.count(PRUNED), states.count(FAIL)) == (14, 6, 6), \
        "expected 14 COMPLETE / 6 PRUNED / 6 FAIL"
    assert study.best_trial.number == 25 and study.best_value == 0.0928155106318572, \
        "best trial must be #25 with value 0.0928155106318572"
    best_row = next(r for r in rows if int(r["trial"]) == 25)
    assert study.best_params == {s["name"]: parse_value(s, best_row[s["name"]]) for s in space}, \
        "best params differ from the CSV row of trial 25"
    return study


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="studies/task4_cgan/trials_from_console_log.csv")
    ap.add_argument("--out-db", default="studies/task4_cgan/task4_cgan.db")
    ap.add_argument("--config", default="configs/task4_cgan.yaml")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing --out-db and rebuild it")
    args = ap.parse_args(argv)

    out_db = resolve(args.out_db)
    study = build_log_study(resolve(args.csv), out_db, resolve(args.config), overwrite=args.overwrite)
    print(f"(trials 0-{N_PLACEHOLDERS - 1} are FAIL placeholders added from the student's recollection; "
          "the console log does not contain them)")
    out_dir = out_db.parent
    export_study(study, out_dir)                          # trials.csv + 3 plots next to the DB
    print("(study rebuilt from the Colab console log, not produced by tune.py)")
    print_study_summary(study, out_dir, source=SOURCE_NOTE)


if __name__ == "__main__":
    main()
