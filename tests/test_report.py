import importlib.util
import json

import pandas as pd
import pytest

from embc import paths, runs
from test_runs import make_stage, write_run

_spec = importlib.util.spec_from_file_location("report", paths.ROOT / "scripts" / "report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


@pytest.fixture
def stage(tmp_path, monkeypatch):
    for name, sub in (("DATA", "data"), ("RUNS", "runs"), ("RESULTS", "results")):
        monkeypatch.setattr(paths, name, tmp_path / sub)
    make_stage(paths.DATA, subjects=("YAC", "YAG"))
    return paths.RUNS / "main" / "fold1"


def _events(subjects, rr):
    return pd.DataFrame(dict(event_id=range(len(subjects)), subject=subjects, RR=rr))


def test_contrast_uses_shared_events_and_flags_partial_coverage():
    a = _events(["A", "A", "B", "B"], [0.5, 0.4, 0.3, 0.2])
    b = _events(["A", "A"], [0.1, 0.1])                                # condition b lacks reader B's fold
    c = report.contrast(a, b)
    assert c["n_events"] == 2 and c["n_subjects"] == 1 and not c["complete"]
    assert report.contrast(a, a)["complete"]
    assert report.contrast(a.iloc[:2], a.iloc[2:]) is None
    g = report.gates({}, {"random_N100": {"raw-noise": c}}, {}, None, None)
    assert g["G1_raw_beats_noise"] is None                              # a gate never uses partial coverage


def test_report_refuses_empty_experiment(stage):
    with pytest.raises(SystemExit, match="no completed runs"):
        report.completed_runs(stage)


def test_report_refuses_stale_and_mixed_protocol_runs(stage):
    write_run(stage / "sf0" / "res" / "seed0")
    write_run(stage / "sf0" / "noise" / "seed0", cond="noise")
    assert len(report.completed_runs(stage)) == 2
    (paths.DATA / "fold1" / "galleries_random_N100.npz").write_text("regenerated galleries")
    with pytest.raises(SystemExit, match="stale runs"):
        report.completed_runs(stage)
    write_run(stage / "sf0" / "res" / "seed0")
    write_run(stage / "sf0" / "noise" / "seed0", cond="noise", epochs=2)
    with pytest.raises(SystemExit, match="mix 2 protocols"):
        report.completed_runs(stage)


def test_probes_count_only_for_current_embeddings(stage):
    d = stage / "sf0" / "res" / "seed0"
    write_run(d)
    (d / "z.npz").write_text("embeddings v1")
    r2 = dict(linear=dict(r2=0.1), mlp=dict(r2=0.2))
    (d / "probes.json").write_text(json.dumps(dict(source=dict(z=runs.file_digest(d / "z.npz")),
                                                   targets=dict(log_ffd=r2))))
    notes = []
    assert len(report.probes_table({d}, notes)) == 1 and not notes
    (d / "z.npz").write_text("embeddings v2, retrained")
    assert report.probes_table({d}, notes) is None and "not computed from its current" in notes[0]


def test_injection_counts_only_for_the_full_current_stage1a(stage):
    paths.RESULTS.mkdir()
    f = paths.RESULTS / "injection_recovery_fold1.csv"
    pd.DataFrame(dict(subject=["YAC"], rho=[0.0], cond=["res"], sig_slope=[1.0])).to_csv(f, index=False)

    def record(subjects):
        meta = dict(fold=1, subjects=subjects, continuous={s: runs.continuous_digest(s) for s in subjects})
        f.with_suffix(".json").write_text(json.dumps(meta))

    notes = []
    record(["YAC"])                                                     # smoke run on one reader
    assert report.injection(1, notes) is None and "covers 1 of 2" in notes[-1]
    record(["YAC", "YAG"])
    assert report.injection(1, notes) is not None
    o = paths.subject_outputs("YAG")                                    # stage 1a rebuilt afterwards
    o["npz"].write_text("YAG rebuilt with another ridge")
    o["info"].write_text(json.dumps(dict(subject="YAG", npz_sha256=runs.file_digest(o["npz"]))))
    assert report.injection(1, notes) is None and "another stage-1a build" in notes[-1]
    assert report.injection(2, notes) is None                           # other folds are not interchangeable
