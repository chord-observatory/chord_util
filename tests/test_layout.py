"""Tests for chord_util.layout."""

import h5py
import numpy as np
import pytest

from chord_util import layout


@pytest.fixture
def xfile(tmp_path):
    fname = tmp_path / "vis.h5"
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [6.300156854906823, 8.500058, 0.0],
            [0.0, 0.0, 0.0],
            [6.300156854906823, 8.500058, 0.0],
        ]
    )
    with h5py.File(fname, "w") as fh:
        fh.attrs["feed_positions_m"] = pos
        fh.attrs["itrs_lat_deg"] = 49.3
        fh.attrs["dish_coelev_deg"] = -27.3
        fh["index_map/label"] = np.array([b"A01X", b"A02X", b"A01Y", b"A02Y"])
        fh["index_map/pol"] = np.array([0, 0, 1, 1])
    return fname


def test_layout_roundtrip_full_precision(xfile, tmp_path):
    out = tmp_path / "layout.txt"
    layout.write_layout(xfile, out)
    lay = layout.read_layout(out)
    assert list(lay["label"]) == ["A01X", "A02X", "A01Y", "A02Y"]
    assert list(lay["pol"]) == ["X", "X", "Y", "Y"]
    assert lay["ew_m"][1] == 6.300156854906823  # exact: no rounding in the file
    assert layout.check_layout(xfile, out) == []


def test_check_layout_finds_a_moved_input(xfile, tmp_path):
    out = tmp_path / "layout.txt"
    layout.write_layout(xfile, out)
    with h5py.File(xfile, "r+") as fh:
        pos = fh.attrs["feed_positions_m"].copy()
        pos[1, 0] += 0.01
        fh.attrs["feed_positions_m"] = pos
    problems = layout.check_layout(xfile, out)
    assert len(problems) == 1 and "A02X" in problems[0]


def test_feedmask_history(tmp_path):
    path = tmp_path / "mask.yaml"
    layout.update_feedmask(
        "2026-10-25T00:00:00Z", {"B01X": "dead"}, acquisition="a", path=path
    )
    layout.update_feedmask(
        "2026-11-06T00:00:00Z",
        {"A02X": "dead", "A02Y": "dead"},
        acquisition="b",
        path=path,
    )
    assert layout.bad_inputs_at("2026-10-30T00:00:00Z", path) == {"B01X": "dead"}
    assert set(layout.bad_inputs_at("2026-11-07T00:00:00Z", path)) == {"A02X", "A02Y"}
    with pytest.raises(ValueError):
        layout.bad_inputs_at("2026-10-01T00:00:00Z", path)
    # Same start time: replaced, not duplicated
    layout.update_feedmask("2026-11-06T00:00:00Z", {"H01X": "hot"}, path=path)
    assert len(layout.read_feedmask(path)) == 2
    assert layout.bad_inputs_at(1794100000.0, path) == {"H01X": "hot"}


def test_pointing_dec():
    assert layout.pointing_dec(
        {"itrs_lat_deg": 49.3, "dish_coelev_deg": -27.3}
    ) == pytest.approx(22.0)
