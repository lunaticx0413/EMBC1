import json

import numpy as np
import pytest

from embc import events, inputs, metrics, paths, runs


def _cfg(tmp_path, **over):
    f = tmp_path / "events.csv"
    if not f.exists():
        f.write_text("event_id\n0\n")
    args = dict(exp="main", fold=1, subject_fold=0, k_subject_folds=3, cond="res", seed=0, epochs=40, patience=6,
                batch=256, lr=1e-3, wd=1e-2, tau=0.07, select_gallery="random_N100", max_train_events=None,
                device="cuda")
    args.update(over)
    return runs.run_config(args, dict(data={"events.csv": runs.file_digest(f)}, code={"embc/model.py": "c0"}))


def make_stage(root, subjects=("YAC",)):
    """Minimal fold-1 inputs and complete stage-1a outputs (built before digests were recorded) under root."""
    fdir = root / "fold1"
    fdir.mkdir(parents=True)
    for name in ("events.csv", "types.npz", "galleries_random_N100.npz"):
        (fdir / name).write_text(name)
    (root / "subject_folds_k3.json").write_text("[]")
    (root / "continuous").mkdir()
    for s in subjects:
        out = paths.subject_outputs(s)
        for k in ("npz", "fix", "words"):
            out[k].write_text(f"{s} {k}")
        out["info"].write_text(json.dumps(dict(subject=s, ridge_rel=1e-3)))


def write_run(out, cond="res", **over):
    out.mkdir(parents=True, exist_ok=True)
    args = dict(exp="main", fold=1, subject_fold=0, k_subject_folds=3, cond=cond, seed=0, **over)
    cfg = runs.run_config(args, runs.input_digests(1, 3, cond))
    runs.write_atomic(out / "metrics.json", json.dumps(dict(config=cfg)))
    return cfg


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    make_stage(paths.DATA)
    return tmp_path


def test_completed_run_is_reused_only_with_identical_config(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    full = _cfg(tmp_path)
    assert runs.existing_run_state(out, full) == "missing"
    smoke = _cfg(tmp_path, epochs=2, max_train_events=3000)
    (out / "metrics.json").write_text(json.dumps({"config": smoke}))
    with pytest.raises(runs.RunConfigMismatch, match="epochs, max_train_events"):
        runs.existing_run_state(out, full)
    assert runs.existing_run_state(out, smoke) == "done"
    assert runs.existing_run_state(out, _cfg(tmp_path, epochs=2, max_train_events=3000, device="cpu")) == "done"


def test_config_diff_names_changed_inputs_and_code(tmp_path):
    a = _cfg(tmp_path)
    (tmp_path / "events.csv").write_text("event_id\n0\n1\n")
    b = _cfg(tmp_path)
    assert runs.config_diff(a, b) == ["data:events.csv"]
    b["code"] = {"embc/model.py": "c1"}
    assert runs.config_diff(a, b) == ["code:embc/model.py", "data:events.csv"]


def test_run_without_recorded_config_is_refused(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"info": {}}))
    with pytest.raises(runs.RunConfigMismatch, match="no recorded configuration"):
        runs.existing_run_state(tmp_path, _cfg(tmp_path))


def test_truncated_metrics_is_a_mismatch_not_a_crash(tmp_path):
    (tmp_path / "metrics.json").write_text('{"config": {"exp": "ma')     # killed mid-write
    with pytest.raises(runs.RunConfigMismatch, match="unreadable"):        # --overwrite handles this one
        runs.existing_run_state(tmp_path, _cfg(tmp_path))
    assert "unreadable" in runs.stale_reason(tmp_path)


def test_write_atomic_replaces_without_leftovers(tmp_path):
    f = tmp_path / "metrics.json"
    f.write_text("old")
    runs.write_atomic(f, "new")
    assert f.read_text() == "new" and [p.name for p in tmp_path.iterdir()] == ["metrics.json"]


def test_input_digests_cover_stage1a_eeg_and_code(stage):
    eeg, gaze, prior = (runs.input_digests(1, 3, c) for c in ("res", "gaze", "prior"))
    assert "continuous/YAC.npz" in eeg["data"]
    assert not any(k.startswith("continuous/") for k in {**gaze["data"], **prior["data"]})
    assert set(eeg["data"]) >= {"events.csv", "types.npz", "galleries_random_N100.npz", "subject_folds_k3.json"}
    assert set(eeg["code"]) == set(runs.CODE_FILES)
    info = json.loads(paths.subject_outputs("YAC")["info"].read_text())
    assert info["npz_sha256"] == eeg["data"]["continuous/YAC.npz"]       # recorded once for older builds


def test_stale_reason_detects_rebuilt_stage1a(stage):
    out = stage / "runs" / "res"
    write_run(out)
    assert runs.stale_reason(out) is None
    o = paths.subject_outputs("YAC")                                      # rebuild with another ridge
    o["npz"].write_text("YAC npz, ridge 1e-2")
    o["info"].write_text(json.dumps(dict(subject="YAC", ridge_rel=1e-2, npz_sha256=runs.file_digest(o["npz"]))))
    assert "data:continuous/YAC.npz" in runs.stale_reason(out)
    prior = stage / "runs" / "prior"
    write_run(prior, cond="prior")
    assert runs.stale_reason(prior) is None                              # prior reads no EEG


def test_stale_reason_detects_code_change(stage, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", stage)
    monkeypatch.setattr(runs, "CODE_FILES", ("train.py",))
    (stage / "train.py").write_text("v1")
    out = stage / "runs" / "res"
    write_run(out)
    assert runs.stale_reason(out) is None
    (stage / "train.py").write_text("v2, partial batches kept")
    assert "code:train.py" in runs.stale_reason(out)


def test_batch_slices_keep_small_and_partial_batches():
    assert runs.batch_slices(100, 256) == [(0, 100)]               # fewer events than one batch still trains
    assert runs.batch_slices(600, 256) == [(0, 256), (256, 512), (512, 600)]
    assert runs.batch_slices(513, 256) == [(0, 256), (256, 512)]   # a 1-event batch has zero InfoNCE gradient
    assert runs.batch_slices(1, 256) == []


def test_selection_gallery_refuses_unknown_and_empty():
    G = np.zeros((0, 100), int)
    gal = {"random_N100": (np.arange(3), np.zeros((3, 100), int)), "balanced_exact_N250": (np.arange(0), G)}
    assert len(runs.selection_gallery(gal, "random_N100")[0]) == 3
    with pytest.raises(SystemExit, match="unknown"):
        runs.selection_gallery(gal, "matched_N100")
    with pytest.raises(SystemExit, match="covers no validation event"):
        runs.selection_gallery(gal, "balanced_exact_N250")


def test_gallery_mrr_uses_the_gallery_rows():
    T = np.eye(3, dtype=np.float32)
    Z = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    G = np.array([[2, 0, 1]])                                       # one gallery, belongs to event row 2
    assert metrics.gallery_mrr(Z, T, np.array([2]), G) == 1.0
    assert metrics.gallery_mrr(Z, T, np.array([0]), G) < 1.0


def test_interrupted_stage1a_reader_is_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    d = tmp_path / "continuous"
    d.mkdir()
    out = paths.subject_outputs("YAC")
    for k in ("npz", "fix", "words"):                               # killed before the info marker
        out[k].write_text("x")
    assert not paths.subject_complete("YAC")
    assert events.subjects_available() == [] and events.incomplete_subjects() == ["YAC"]
    out["info"].write_text("{}")
    assert paths.subject_complete("YAC") and events.subjects_available() == ["YAC"]
    assert events.incomplete_subjects() == []


def test_stray_files_are_not_readers(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    d = tmp_path / "continuous"
    d.mkdir()
    for k, f in paths.subject_outputs("YAC").items():
        f.write_text("x")
    for name in (".directory", ".DS_Store", "notes.txt", "YAC_info.json.bak"):
        (d / name).write_text("x")
    assert events.incomplete_subjects() == [] and events.subjects_available() == ["YAC"]
    for name in ("YAG.npz.tmp.npz", "YAH_fix.csv.tmp", "YAK_info.json.tmp4242"):   # interrupted writes
        (d / name).write_text("x")
    assert events.incomplete_subjects() == ["YAG", "YAH", "YAK"]


def test_subject_rng_differs_between_readers():
    a, b = (inputs.subject_rng(s, 0).standard_normal(5) for s in ("YAC", "YAG"))
    assert not np.allclose(a, b)
    assert np.allclose(a, inputs.subject_rng("YAC", 0).standard_normal(5))
