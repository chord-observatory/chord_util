"""Tests for chord_util.andata, using synthetic X-engine files."""

import json
import os

import h5py
import numpy as np
import pytest

from chord_util import andata, testing


@pytest.fixture(scope="module")
def xfiles(tmp_path_factory):
    """Synthetic files (with the CRS board remap recorded) and their truth."""
    return testing.make_xengine_files(tmp_path_factory.mktemp("acq"))


@pytest.fixture(scope="module")
def files(xfiles):
    return xfiles[0]


@pytest.fixture(scope="module")
def truth(xfiles):
    return xfiles[1]


@pytest.fixture(scope="module")
def data(files):
    """All the files read with the default options."""
    return andata.CorrData.from_acq_h5(files)


def _true_vis(truth, data):
    """The truth in the units of the loaded data."""
    return truth["vis"] * data.attrs["digital_gain_norm"] ** 2


def _true_weight(truth, data):
    w = truth["weight"] / data.attrs["digital_gain_norm"] ** 4
    return np.where(truth["missing"][:, np.newaxis, :], 0.0, w)


# Functions
# ---------


def test_digital_gain_input():
    """Element e is fed by lane e % 8 of board crs_board_remap[e // 8]."""
    fin = andata.digital_gain_input([0, 7, 8, 15, 56, 63, 64, 71, 72, 120, 127])
    assert fin.tolist() == [0, 7, 16, 23, 120, 127, 8, 15, 24, 112, 119]

    identity = list(range(16))
    assert andata.digital_gain_input(np.arange(128), identity).tolist() == list(
        range(128)
    )


def test_get_crs_board_remap(files, tmp_path):
    with h5py.File(files[0], "r") as fh:
        assert andata.get_crs_board_remap(fh) == list(testing.TEST_CRS_BOARD_REMAP)

    # Not recorded
    fname = tmp_path / "noremap.h5"
    with h5py.File(fname, "w") as fh:
        fh["config_json"] = np.array(
            [json.dumps({"config": {}})], dtype=h5py.string_dtype()
        )
    with h5py.File(fname, "r") as fh:
        assert andata.get_crs_board_remap(fh) is None

    # Snapshots that disagree
    with h5py.File(fname, "w") as fh:
        snaps = [
            json.dumps({"config": {"dpdk": {"crs_board_remap": list(range(16))}}}),
            json.dumps(
                {"config": {"dpdk": {"crs_board_remap": list(range(16))[::-1]}}}
            ),
        ]
        fh["config_json"] = np.array(snaps, dtype=h5py.string_dtype())
    with h5py.File(fname, "r") as fh, pytest.raises(andata.AnDataError):
        andata.get_crs_board_remap(fh)


def test_read_digital_gains_ignores_exp(files, truth):
    """The gain is gain_coeff alone, gain_exp is ignored."""
    fin = andata.digital_gain_input(truth["input_list"], testing.TEST_CRS_BOARD_REMAP)
    with h5py.File(files[0], "r") as fh:
        g = andata.read_digital_gains(fh, truth["freq"], fin)
        assert np.all(fh["digital_gains/gain_exp"][0][fin] == -1)
    np.testing.assert_allclose(g, truth["gain"][0])


def test_versiontuple():
    assert andata.versiontuple("1.2.3") == (1, 2, 3)
    assert andata.versiontuple("CHORD_0.0") == (0, 0)


def test_subclass_from_obj(files, data):
    assert andata.subclass_from_obj(andata.BaseData, files[0]) is andata.CorrData
    assert andata.subclass_from_obj(andata.BaseData, data) is andata.CorrData
    assert andata.subclass_from_obj(andata.BaseData, None) is andata.BaseData


# Reading
# -------


def test_shape_and_axes(data, truth):
    nfreq, nprod, ntime = truth["vis"].shape
    assert data.vis.shape == (nfreq, nprod, ntime)
    assert data.weight.shape == (nfreq, nprod, ntime)
    assert list(data.vis.attrs["axis"]) == ["freq", "prod", "time"]
    assert data.nfreq == nfreq and data.nprod == nprod and data.ntime == ntime
    assert data.ninput == len(truth["labels"])
    assert data.nstack == nprod and not data.is_stacked
    assert np.array_equal(data.prodstack, data.prod)


def test_unfilled_bins_dropped_and_times(data, truth):
    """Unfilled time bins are dropped and the time axis is the UNIX time."""
    np.testing.assert_allclose(data.time, truth["time"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(data.timestamp, data.time)
    assert np.all(np.diff(data.time) > 0)
    np.testing.assert_array_equal(data["file_index"][:], truth["file_index"])
    np.testing.assert_allclose(
        data["time_center_t_inst_ns"][:] * 1e-9, data.time, atol=1e-6
    )


def test_gains_divided_out(data, truth):
    """vis and weights are the truth, up to the gain normalisation."""
    assert data.attrs["digital_gains_applied"]
    np.testing.assert_allclose(data.attrs["digital_gain_norm"], truth["gain_norm"])
    np.testing.assert_allclose(data.vis[:], _true_vis(truth, data), rtol=1e-5)
    np.testing.assert_allclose(data.weight[:], _true_weight(truth, data), rtol=1e-4)
    np.testing.assert_allclose(
        data.digital_gain[:], truth["gain"][0] / truth["gain_norm"], rtol=1e-6
    )


def test_crs_board_remap_used(data, truth):
    assert data.attrs["crs_board_remap_source"] == "config_json"
    assert list(data.attrs["crs_board_remap"]) == truth["crs_board_remap"]


def test_missing_samples_zero_weight(data, truth):
    fi, ti = np.nonzero(truth["missing"])
    assert np.all(data.weight[:][fi, :, ti] == 0)


def test_flags(data, truth):
    np.testing.assert_array_equal(
        data.flags["element_flags"][:], truth["element_flags"]
    )
    np.testing.assert_array_equal(data.input_flags[:], truth["input_flags"])
    for name in andata.FREQ_TIME_FLAGS:
        assert data.flags[name].shape == (data.nfreq, data.ntime)
    np.testing.assert_array_equal(
        data.frac_lost[:] >= 1, truth["missing"] & (data.frac_lost[:] >= 1)
    )


def test_index_maps(data, truth):
    np.testing.assert_allclose(data.freq, truth["freq"])
    assert data.prod.tolist() == truth["prod"].tolist()
    assert data.input["chan_id"].tolist() == truth["input_list"].tolist()
    assert data.labels.tolist() == truth["labels"]
    assert data.stack["prod"].tolist() == list(range(data.nprod))
    assert data.reverse_map["stack"]["stack"].tolist() == list(range(data.nprod))
    assert len(data.index_map["file"]) == 3


def test_input_info(data, truth):
    np.testing.assert_array_equal(data.input_type, truth["input_type"])
    np.testing.assert_array_equal(
        data.is_dish, truth["input_type"] == andata.INPUT_TYPE_DISH
    )
    np.testing.assert_array_equal(
        data.is_rfi_monitor, truth["input_type"] == andata.INPUT_TYPE_RFI
    )
    for name in andata.INPUT_TABLES + andata.INPUT_ATTRS:
        assert data.input_info[name].shape[0] == data.ninput


def test_default_datasets(data):
    """Placeholder datasets are not read by default, everything else is."""
    for name in andata.PLACEHOLDER_DATASETS:
        assert name not in data
    for name in ["vis", "eval", "evec", "erms", "digital_gain", "config_json"]:
        assert name in data.datasets
    for name in andata.TIME_DATASETS:
        assert name in data.datasets
    assert "bf_mask" not in data._data


def test_attributes(data, files):
    assert data.attrs["acquisition_type"] == "corr"
    assert data.attrs["acquisition"] == os.path.basename(os.path.dirname(files[0]))
    assert data.attrs["num_elements"] == data.ninput
    assert data.attrs["num_prod"] == data.nprod
    assert data.attrs["num_file_f"] == data.nfreq
    assert "input_list" not in data.attrs


def test_file_order_and_input_forms(files, data):
    """Files are sorted, and may be given as a glob or open h5py files."""
    shuffled = andata.CorrData.from_acq_h5([files[2], files[0], files[1]])
    np.testing.assert_array_equal(shuffled.vis[:], data.vis[:])

    pattern = os.path.join(os.path.dirname(files[0]), "vis_*.h5")
    np.testing.assert_array_equal(
        andata.CorrData.from_acq_h5(pattern).vis[:], data.vis[:]
    )

    with h5py.File(files[0], "r") as f0, h5py.File(files[1], "r") as f1:
        two = andata.CorrData.from_acq_h5([f0, f1])
    assert two.ntime < data.ntime


def test_no_gain_correction(files, truth):
    raw = andata.CorrData.from_acq_h5(files, apply_gain=False)
    assert not raw.attrs["digital_gains_applied"]
    g = truth["gain"][0]
    gg = g[:, truth["prod"]["input_a"]] * g[:, truth["prod"]["input_b"]].conj()
    np.testing.assert_allclose(raw.vis[:], gg[..., None] * truth["vis"], rtol=1e-5)


def test_unnormalised_gains(files, truth):
    data = andata.CorrData.from_acq_h5(files, normalise_gain=False)
    assert data.attrs["digital_gain_norm"] == 1.0
    np.testing.assert_allclose(data.vis[:], truth["vis"], rtol=1e-5)


def test_remap_default_and_argument(tmp_path):
    """Without a recorded remap the default is used; an argument overrides it."""
    files, truth = testing.make_xengine_files(
        tmp_path, crs_board_remap=andata.CRS_BOARD_REMAP, record_remap=False, nfile=2
    )
    data = andata.CorrData.from_acq_h5(files)
    assert data.attrs["crs_board_remap_source"] == "default"
    np.testing.assert_allclose(data.vis[:], _true_vis(truth, data), rtol=1e-5)

    wrong = andata.CorrData.from_acq_h5(files, crs_board_remap=list(range(16)))
    assert wrong.attrs["crs_board_remap_source"] == "argument"
    assert not np.allclose(wrong.vis[:], _true_vis(truth, wrong), rtol=1e-3)


def test_zero_gain_input(tmp_path):
    """Products of an input with a zero gain get zero weight."""
    files, truth = testing.make_xengine_files(tmp_path, zero_gain_input=1, nfile=2)
    data = andata.CorrData.from_acq_h5(files)
    involved = (truth["prod"]["input_a"] == 1) | (truth["prod"]["input_b"] == 1)
    assert np.all(data.weight[:][:, involved] == 0)
    others = data.weight[:][:, ~involved].transpose(0, 2, 1)  # (freq, time, prod)
    assert np.all(others[~truth["missing"]] > 0)


def test_gains_change_between_files(tmp_path):
    """Each file is corrected with its own gains."""
    files, truth = testing.make_xengine_files(tmp_path, change_gains=True)
    data = andata.CorrData.from_acq_h5(files)
    np.testing.assert_allclose(data.vis[:], _true_vis(truth, data), rtol=1e-5)


def test_layout_mismatch(files, tmp_path):
    other, _ = testing.make_xengine_files(
        tmp_path, nfreq=8, nfile=1, acquisition="acq_other"
    )
    with pytest.raises(andata.AnDataError):
        andata.CorrData.from_acq_h5([files[0], other[0]])


def test_not_xengine_file(tmp_path):
    fname = tmp_path / "vis_0000000001_20260101T_000000_000000000.h5"
    with h5py.File(fname, "w") as fh:
        fh.attrs["file_mode"] = "CHIME"
    with pytest.raises(andata.AnDataError):
        andata.CorrData.from_acq_h5(str(fname))


def test_unknown_kwarg(files):
    with pytest.raises(ValueError):
        andata.CorrData.from_acq_h5(files, not_an_option=True)
    # Accepted for ch_util compatibility, and ignored
    andata.CorrData.from_acq_h5(files[:1], renormalize=False)


# Selections
# ----------


@pytest.mark.parametrize(
    "freq_sel",
    [slice(2, 10), slice(1, 15, 3), [0, 4, 5, 11], np.arange(16) % 2 == 0, 7],
)
def test_freq_sel(files, data, freq_sel):
    sub = andata.CorrData.from_acq_h5(files, freq_sel=freq_sel)
    ind = np.arange(data.nfreq)[andata._ensure_1D_selection(freq_sel)]
    np.testing.assert_array_equal(sub.vis[:], data.vis[:][ind])
    np.testing.assert_array_equal(sub.weight[:], data.weight[:][ind])
    np.testing.assert_array_equal(sub.digital_gain[:], data.digital_gain[:][ind])
    np.testing.assert_array_equal(sub.freq, data.freq[ind])


@pytest.mark.parametrize("start, stop", [(0, 5), (3, None), (None, -2), (4, 9)])
def test_time_range(files, data, start, stop):
    sub = andata.CorrData.from_acq_h5(files, start=start, stop=stop)
    np.testing.assert_array_equal(sub.vis[:], data.vis[:][..., start:stop])
    np.testing.assert_array_equal(sub.time, data.time[start:stop])
    np.testing.assert_array_equal(
        sub["bin_ERA_deg"][:], data["bin_ERA_deg"][start:stop]
    )


def test_input_sel(files, data, truth):
    """Selecting inputs keeps the products between them, re-indexed."""
    isel = np.flatnonzero(truth["input_type"] == andata.INPUT_TYPE_DISH)
    sub = andata.CorrData.from_acq_h5(files, input_sel=isel)
    assert sub.ninput == len(isel) and sub.nprod == len(isel) * (len(isel) + 1) // 2
    assert sub.labels.tolist() == [truth["labels"][i] for i in isel]
    assert not sub.is_rfi_monitor.any()

    labels = data.labels.tolist()
    index = {(labels[a], labels[b]): i for i, (a, b) in enumerate(data.prod)}
    sub_labels = sub.labels.tolist()
    pmap = [index[(sub_labels[a], sub_labels[b])] for a, b in sub.prod]
    np.testing.assert_array_equal(sub.vis[:], data.vis[:][:, pmap])
    np.testing.assert_array_equal(sub.weight[:], data.weight[:][:, pmap])
    np.testing.assert_array_equal(
        sub.flags["element_flags"][:], data.flags["element_flags"][:][:, isel]
    )
    np.testing.assert_array_equal(sub.digital_gain[:], data.digital_gain[:][:, isel])


def test_prod_and_stack_sel(files, data):
    autos = np.flatnonzero(data.prod["input_a"] == data.prod["input_b"])
    sub = andata.CorrData.from_acq_h5(files, prod_sel=autos)
    np.testing.assert_array_equal(sub.vis[:], data.vis[:][:, autos])
    assert np.all(sub.prod["input_a"] == sub.prod["input_b"])
    assert sub.ninput == data.ninput

    sub2 = andata.CorrData.from_acq_h5(files, stack_sel=autos)
    np.testing.assert_array_equal(sub2.vis[:], sub.vis[:])

    with pytest.raises(ValueError):
        andata.CorrData.from_acq_h5(files, prod_sel=autos, input_sel=[0, 1])


def test_datasets_sel(files):
    sub = andata.CorrData.from_acq_h5(files, datasets=["vis", "vis_weight", "gain"])
    assert (
        "vis" in sub.datasets and "vis_weight" in sub.flags and "gain" in sub.datasets
    )
    assert np.all(sub.gain[:] == -1)
    assert "eval" not in sub and "flags/frac_lost" not in sub
    # The per-time and per-input datasets are always read
    assert "bin_ERA_deg" in sub.datasets and "type" in sub.input_info
    # Datasets not in the files are skipped
    sub2 = andata.CorrData.from_acq_h5(files, datasets=["vis", "not_a_dataset"])
    assert "vis" in sub2.datasets


def test_save_and_load(data, tmp_path):
    fname = str(tmp_path / "corrdata.h5")
    data.save(fname)
    back = andata.CorrData.from_file(fname)
    assert isinstance(back, andata.CorrData)
    np.testing.assert_array_equal(back.vis[:], data.vis[:])
    np.testing.assert_array_equal(back.weight[:], data.weight[:])
    np.testing.assert_array_equal(
        back.input_info["type"][:], data.input_info["type"][:]
    )

    # BaseData.from_acq_h5 concatenates files already in analysis format
    cat = andata.BaseData.from_acq_h5([fname, fname])
    assert isinstance(cat, andata.CorrData)
    assert cat.vis.shape[-1] == 2 * data.ntime
    np.testing.assert_array_equal(cat.vis[:][..., data.ntime :], data.vis[:])


# CorrReader
# ----------


@pytest.fixture
def reader(files):
    return andata.CorrReader(files)


def test_reader_metadata(reader, data):
    assert len(reader.files) == 3
    np.testing.assert_allclose(reader.time, data.time)
    assert reader.prod.tolist() == data.prod.tolist()
    assert reader.input.tolist() == data.input.tolist()
    np.testing.assert_array_equal(reader.freq["centre"], data.freq)
    assert "vis" in reader.datasets and "gain" in reader.datasets
    assert "gain" not in reader.dataset_sel


def test_reader_read_all(reader, data):
    np.testing.assert_array_equal(reader.read().vis[:], data.vis[:])


def test_reader_time(reader, data):
    reader.time_sel = (2, 7)
    np.testing.assert_array_equal(reader.read().vis[:], data.vis[:][..., 2:7])

    reader.time_sel = (0, data.ntime)
    reader.select_time_range(data.time[3] - 0.1, data.time[8] + 0.1)
    assert reader.time_sel == (3, 9)


def test_reader_freq(reader, data):
    reader.select_freq_range(data.freq[2] - 0.01, data.freq[9] + 0.01)
    np.testing.assert_array_equal(reader.read().freq, data.freq[2:10])

    reader.select_freq_physical([data.freq[5] + 0.05, data.freq[1]])
    np.testing.assert_array_equal(reader.read().freq, data.freq[[1, 5]])

    with pytest.raises(ValueError):
        reader.select_freq_physical([100.0])


def test_reader_inputs_and_prods(reader, data, truth):
    reader.select_input_type("dish")
    dish = reader.read()
    assert dish.labels.tolist() == [
        lbl for lbl, t in zip(truth["labels"], truth["input_type"]) if t == 0
    ]

    reader.select_input_type("rfi")
    assert reader.read().is_rfi_monitor.all()

    reader.select_input_labels(["B3p2", "B4p1"])
    assert reader.read().labels.tolist() == ["B4p1", "B3p2"]
    with pytest.raises(ValueError):
        reader.select_input_labels(["nope"])

    reader.select_prod_autos()
    assert reader.input_sel is None
    autos = reader.read()
    assert np.all(autos.prod["input_a"] == autos.prod["input_b"])

    reader.select_prod_pairs([(1, 0), (2, 2)])
    pairs = reader.read()
    assert pairs.nprod == 2

    reader.select_prod_by_input(2)
    sub = reader.read()
    assert sub.nprod == data.ninput

    with pytest.raises(ValueError):
        reader.input_sel = [0, 1]


def test_reader_datasets(reader):
    reader.dataset_sel = ["vis"]
    sub = reader.read()
    assert "vis" in sub.datasets and "vis_weight" not in sub.flags
    with pytest.raises(ValueError):
        reader.dataset_sel = ["not_a_dataset"]


# Distributed
# -----------


def test_distributed_matches_serial(files, data):
    """With any number of MPI ranks the distributed read gathers to the serial one."""
    dist = andata.CorrData.from_acq_h5(files, distributed=True)
    assert dist.vis.distributed
    np.testing.assert_array_equal(dist.vis[:].allgather(), data.vis[:])
    np.testing.assert_array_equal(dist.weight[:].allgather(), data.weight[:])
    np.testing.assert_array_equal(dist.input_flags[:], data.input_flags[:])
    np.testing.assert_array_equal(dist.freq, data.freq)
    assert dist.attrs["num_file_f"] == data.nfreq

    fast = andata.CorrData.from_acq_h5_fast(files, freq_sel=slice(3, 11))
    np.testing.assert_array_equal(fast.vis[:].allgather(), data.vis[:][3:11])
    with pytest.raises(ValueError):
        andata.CorrData.from_acq_h5_fast(files, freq_sel=[1, 2])
