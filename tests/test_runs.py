import json

import numpy as np
import pytest

from embc import events, metrics, paths, runs


def _cfg(tmp_path, **over):
    f = tmp_path / "events.csv"
    f.write_text("event_id\n0\n")
    args = dict(exp="main", fold=1, subject_fold=0, k_subject_folds=3, cond="res", seed=0, epochs=40, patience=6,
                batch=256, lr=1e-3, wd=1e-2, tau=0.07, select_gallery="random_N100", max_train_events=None,
                device="cuda")
    args.update(over)
    return runs.run_config(args, [f])


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


def test_config_tracks_input_data_content(tmp_path):
    a = _cfg(tmp_path)
    (tmp_path / "events.csv").write_text("event_id\n0\n1\n")
    assert runs.run_config(dict(exp="main"), [tmp_path / "events.csv"])["data"] != a["data"]


def test_run_without_recorded_config_is_refused(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"info": {}}))
    with pytest.raises(runs.RunConfigMismatch, match="no recorded configuration"):
        runs.existing_run_state(tmp_path, _cfg(tmp_path))


def test_batch_slices_keep_small_and_partial_batches():
    assert runs.batch_slices(100, 256) == [(0, 100)]               # fewer events than one batch still trains
    assert runs.batch_slices(600, 256) == [(0, 256), (256, 512), (512, 600)]
    assert runs.batch_slices(513, 256) == [(0, 256), (256, 512)]   # a 1-event batch would break BatchNorm
    assert runs.batch_slices(1, 256) == []


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
