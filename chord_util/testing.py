"""Tools for testing code that reads CHORD X-engine data.

:func:`make_xengine_files` writes small, synthetic X-engine files with the same
layout as the files from kotekan's ``hdf5N2Write`` (CHORD mode), together with
the "true" data they were made from, so that readers can be checked against a
known answer without access to real data.

Examples
--------
>>> from chord_util import andata, testing
>>> files, truth = testing.make_xengine_files(tmp_path)
>>> data = andata.CorrData.from_acq_h5(files)
>>> np.allclose(data.vis[:] / data.attrs["digital_gain_norm"] ** 2, truth["vis"])
True
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import h5py
import numpy as np
from bitshuffle import h5 as bshufh5

from .andata import CRS_BOARD_REMAP, INPUT_TYPE_DISH

# Channel width and index of the first science channel, as for the pathfinder
CHANNEL_WIDTH_MHZ = 3200.0 / 16384
FIRST_FREQ_ID = 1536

# Default set of inputs: two dishes and one RFI monitor, in both polarisations,
# in the file (CHORDBeamformer) order: pol 0 dishes, pol 0 RFI, pol 1 ...
DEFAULT_INPUTS = {
    "input_list": [0, 1, 56, 64, 65, 120],
    "label": ["B4p1", "B3p1", "RFIB4p1", "B4p2", "B3p2", "RFIB4p2"],
    "type": [0, 0, 1, 0, 0, 1],
    "pol": [0, 0, 0, 1, 1, 1],
    "dish_idx": [0, 1, 56, 0, 1, 56],
}

# A remap that is not the identity for any of the default inputs, so a reader
# that ignores it gets the wrong gains
TEST_CRS_BOARD_REMAP = (3, 5, 0, 2, 4, 6, 8, 15, 7, 1, 9, 11, 13, 10, 12, 14)

# Number of gain archive channels and correlator inputs in the synthetic files
N_GAIN_FREQ = 2048
N_GAIN_INPUT = 128


def _fengine_input(input_list, crs_board_remap):
    """F-engine input feeding each element.

    Deliberately written independently of `andata.digital_gain_input`, so that the
    synthetic data does not inherit bugs from the code it is used to test: element
    ``e`` is ADC lane ``e % 8`` of the board in output slot ``e // 8``.
    """
    out = []
    for element in input_list:
        slot, lane = element // 8, element % 8
        board = crs_board_remap[slot]
        out.append(board * 8 + lane)
    return np.array(out)


# ERA bin length (s) and a start time for the synthetic data
BIN_S = 9.98244352
T0_UNIX_NS = 1789189194185533750
J2000_UNIX_S = 946728000.0


def make_xengine_files(
    directory,
    nfile=3,
    ntime=5,
    nfreq=16,
    unfilled_first=2,
    unfilled_last=1,
    crs_board_remap=TEST_CRS_BOARD_REMAP,
    record_remap=True,
    zero_gain_input=None,
    change_gains=False,
    acquisition="acq_20260911_232919_986789057",
    seed=1234,
):
    """Write synthetic X-engine files and return the data they encode.

    The true visibilities and weights are random. The raw files hold them with
    the digital gains applied, ``V_ab = g_a g_b^* V_true``,
    ``w_ab = w_true / |g_a g_b|^2``, where the gain of each element is taken from
    the gain archive through `crs_board_remap`.

    Parameters
    ----------
    directory : str or path
        Directory in which the acquisition directory is created.
    nfile : int
        Number of files.
    ntime : int
        Time bins per file.
    nfreq : int
        Frequency channels per file.
    unfilled_first, unfilled_last : int
        Number of unfilled time bins at the start of the first file and at the end
        of the last file (as at the start and end of real acquisitions).
    crs_board_remap : sequence of int
        The CRS board remap used to apply the gains.
    record_remap : bool
        Record `crs_board_remap` in the ``/config_json`` receiver snapshot.
    zero_gain_input : int, optional
        Index of an input whose gain is set to zero.
    change_gains : bool
        Use a different gain update in the last file.
    acquisition : str
        Name of the acquisition directory.
    seed : int
        Random seed.

    Returns
    -------
    files : list of str
        The files, in time order.
    truth : dict
        ``vis`` (freq, prod, time) and ``weight`` true data for the filled samples;
        ``time`` (UNIX s); ``freq`` (MHz); ``prod``; ``labels``; ``input_type``;
        ``gain`` (file, freq, input) applied gains; ``gain_norm`` expected
        normalisation; ``missing`` (freq, time) samples with no data;
        ``element_flags`` (freq, input, time); ``input_flags`` (input, time);
        ``file_index`` (time); ``crs_board_remap``.
    """
    rng = np.random.default_rng(seed)
    acq_dir = os.path.join(str(directory), acquisition)
    os.makedirs(acq_dir, exist_ok=True)

    inputs = DEFAULT_INPUTS
    ninput = len(inputs["input_list"])
    prod = np.array(
        [(a, b) for a in range(ninput) for b in range(a, ninput)],
        dtype=[("input_a", "<u2"), ("input_b", "<u2")],
    )
    nprod = len(prod)
    auto = prod["input_a"] == prod["input_b"]

    freq_id = FIRST_FREQ_ID + np.arange(nfreq)
    freq = np.empty(nfreq, dtype=[("centre", "<f8"), ("width", "<f8")])
    freq["centre"] = freq_id * CHANNEL_WIDTH_MHZ
    freq["width"] = CHANNEL_WIDTH_MHZ

    # Which time bins are filled
    ntot = nfile * ntime
    filled = np.ones(ntot, dtype=bool)
    filled[:unfilled_first] = False
    if unfilled_last:
        filled[-unfilled_last:] = False
    nfilled = int(filled.sum())

    # True data for the filled samples
    vis_true = (
        rng.normal(size=(nfreq, nprod, nfilled))
        + 1j * rng.normal(size=(nfreq, nprod, nfilled))
    ).astype(np.complex64)
    vis_true[:, auto] = rng.uniform(5, 20, size=(nfreq, auto.sum(), nfilled))
    weight_true = rng.uniform(0.5, 2.0, size=(nfreq, nprod, nfilled)).astype(np.float32)

    # Samples with no data: one never received, one fully excised
    missing = np.zeros((nfreq, nfilled), dtype=bool)
    frames_added = np.ones((nfreq, nfilled), dtype=np.uint8)
    frac_lost = rng.uniform(0, 0.1, size=(nfreq, nfilled)).astype(np.float32)
    received, excised = (3 % nfreq, 1 % nfilled), (5 % nfreq, 2 % nfilled)
    frames_added[received] = 0
    frac_lost[excised] = 1.0
    missing[received] = missing[excised] = True

    # X-engine element flags: RFI monitors flagged bad, one dish input flagged
    # bad for one sample at one frequency
    element_flags = np.ones((nfreq, ninput, nfilled), dtype=np.float32)
    element_flags[:, np.array(inputs["type"]) != INPUT_TYPE_DISH] = 0.0
    element_flags[2 % nfreq, 1, 3 % nfilled] = 0.0

    # Gain archive(s)
    fengine_input = _fengine_input(inputs["input_list"], crs_board_remap)
    gain_freq = np.empty(N_GAIN_FREQ, dtype=[("centre", "<f8"), ("width", "<f8")])
    gain_freq["centre"] = np.arange(N_GAIN_FREQ) * CHANNEL_WIDTH_MHZ
    gain_freq["width"] = CHANNEL_WIDTH_MHZ

    def _gain_archive():
        coeff = (
            rng.uniform(0.5e3, 2e3, size=(N_GAIN_FREQ, N_GAIN_INPUT))
            * np.exp(2j * np.pi * rng.uniform(size=(N_GAIN_FREQ, N_GAIN_INPUT)))
        ).astype(np.complex64)
        if zero_gain_input is not None:
            coeff[:, fengine_input[zero_gain_input]] = 0.0
        return coeff

    archives = [_gain_archive()]
    if change_gains:
        archives.append(_gain_archive())
    archive_of_file = [0] * nfile
    if change_gains:
        archive_of_file[-1] = 1

    applied = np.array(
        [
            archives[a][freq_id][:, fengine_input].astype(np.complex128)
            for a in archive_of_file
        ]
    )
    dish = np.array(inputs["type"]) == INPUT_TYPE_DISH
    g0 = np.abs(applied[0][:, dish])
    gain_norm = float(np.median(g0[g0 > 0]))

    # Times
    t_inst_ns = T0_UNIX_NS + np.round(np.arange(ntot) * BIN_S * 1e9).astype(np.int64)
    t_ut1_ns = t_inst_ns - int(J2000_UNIX_S * 1e9)
    bin_abs = 84483820 + np.arange(ntot, dtype=np.uint64)

    files = []
    it = 0  # index into the filled samples
    for fi in range(nfile):
        tsl = slice(fi * ntime, (fi + 1) * ntime)
        fill = filled[tsl]
        nf = int(fill.sum())
        fsl = slice(it, it + nf)
        it += nf

        abs_file_idx = 4224075 + fi
        start = datetime.fromtimestamp(
            t_inst_ns[tsl][fill][0] * 1e-9 if nf else 0, tz=timezone.utc
        )
        fname = os.path.join(
            acq_dir,
            f"vis_{abs_file_idx:010d}_{start:%Y%m%d}T_{start:%H%M%S}_{start.microsecond * 1000:09d}.h5",
        )

        g = applied[fi]
        gg = g[:, prod["input_a"]] * g[:, prod["input_b"]].conj()

        def _full(arr, fill_value=0):
            """Put per-filled-sample data into the file's time bins."""
            out = np.full(arr.shape[:-1] + (ntime,), fill_value, dtype=arr.dtype)
            out[..., fill] = arr
            return out

        raw_vis = _full((gg[..., None] * vis_true[..., fsl]).astype(np.complex64))
        with np.errstate(divide="ignore"):
            raw_w = np.where(
                np.abs(gg[..., None]) > 0,
                weight_true[..., fsl] / np.abs(gg[..., None]) ** 2,
                0.0,
            ).astype(np.float32)
        raw_w = _full(raw_w)

        with h5py.File(fname, "w") as fh:
            _write_file(
                fh,
                fi=fi,
                abs_file_idx=abs_file_idx,
                inputs=inputs,
                prod=prod,
                freq=freq,
                vis=raw_vis,
                weight=raw_w,
                frames_added=_full(frames_added[:, fsl]),
                frac_lost=_full(frac_lost[:, fsl], 1.0),
                element_flags=_full(element_flags[..., fsl], 1.0),
                fill=fill,
                t_inst_ns=np.where(fill, t_inst_ns[tsl], 0),
                t_ut1_ns=np.where(fill, t_ut1_ns[tsl], 0),
                bin_abs=np.where(fill, bin_abs[tsl], np.iinfo(np.uint64).max),
                gain_coeff=archives[archive_of_file[fi]],
                gain_freq=gain_freq,
                update=archive_of_file[fi],
                remap=crs_board_remap if record_remap else None,
            )
        files.append(fname)

    truth = {
        "vis": vis_true,
        "weight": weight_true,
        "time": t_inst_ns[filled] * 1e-9,
        "freq": freq["centre"],
        "prod": prod,
        "labels": list(inputs["label"]),
        "input_type": np.array(inputs["type"]),
        "input_list": np.array(inputs["input_list"]),
        "gain": applied,
        "gain_norm": gain_norm,
        "missing": missing,
        "element_flags": element_flags,
        "input_flags": (element_flags > 0).any(axis=0).astype(np.float32),
        "file_index": np.repeat(np.arange(nfile), ntime)[filled],
        "crs_board_remap": list(crs_board_remap),
    }
    return files, truth


def _write_file(
    fh,
    fi,
    abs_file_idx,
    inputs,
    prod,
    freq,
    vis,
    weight,
    frames_added,
    frac_lost,
    element_flags,
    fill,
    t_inst_ns,
    t_ut1_ns,
    bin_abs,
    gain_coeff,
    gain_freq,
    update,
    remap,
):
    """Write one file in the kotekan ``hdf5N2Write`` CHORD layout."""
    nfreq, nprod, ntime = vis.shape
    ninput = len(inputs["input_list"])
    rng = np.random.default_rng(fi)

    def dset(name, data, axes, compress=False):
        kw = {}
        if compress:
            kw = {
                "compression": bshufh5.H5FILTER,
                "compression_opts": (0, bshufh5.H5_COMPRESS_LZ4),
                "chunks": True,
            }
        d = fh.create_dataset(name, data=data, **kw)
        d.attrs["axis"] = np.array(axes, dtype=object)
        return d

    # Root attributes
    fh.attrs["version"] = "CHORD_0.0"
    fh.attrs["file_mode"] = "CHORD"
    fh.attrs["abs_file_idx"] = abs_file_idx
    fh.attrs["num_file_t"] = ntime
    fh.attrs["num_file_f"] = nfreq
    fh.attrs["num_telescope_f"] = 8192
    fh.attrs["num_elements"] = ninput
    fh.attrs["num_prod"] = nprod
    fh.attrs["num_ev"] = 2
    fh.attrs["n2_layout"] = "DishInputs"
    fh.attrs["input_order"] = "CHORDBeamformer"
    fh.attrs["input_list"] = np.array(inputs["input_list"], dtype=np.int32)
    fh.attrs["num_dishes"] = 64
    fh.attrs["instrument_name"] = "CHORDTelescope"
    fh.attrs["fpga_seq_length_ns"] = 5120
    fh.attrs["itrs_lat_deg"] = 49.32075144444
    fh.attrs["itrs_lon_deg"] = -119.62081125
    fh.attrs["dish_coelev_deg"] = -27.3
    fh.attrs["feed_positions_m"] = rng.uniform(0, 50, size=(ninput, 3))
    fh.attrs["main_array_grid_indices"] = np.array(
        [
            [d % 8, d // 8] if t == 0 else [-1, -1]
            for d, t in zip(inputs["dish_idx"], inputs["type"])
        ]
    )
    fh.attrs["digital_gains_source_file"] = "baseband_gains.h5"

    # Index maps
    im = fh.create_group("index_map")
    dset("index_map/freq", freq, ["frequency"])
    dset("index_map/prod", prod, ["product"])
    dset(
        "index_map/label",
        np.array(inputs["label"], dtype=h5py.string_dtype()),
        ["element"],
    )
    dset("index_map/type", np.array(inputs["type"], dtype=np.int32), ["element"])
    dset("index_map/pol", np.array(inputs["pol"], dtype=np.int32), ["element"])
    dset(
        "index_map/dish_idx", np.array(inputs["dish_idx"], dtype=np.int64), ["element"]
    )
    dset("index_map/grid_x_idx", np.array(inputs["dish_idx"]) % 8, ["element"])
    dset("index_map/grid_y_idx", np.array(inputs["dish_idx"]) // 8, ["element"])
    dset("index_map/coelev_disp_deg", np.zeros(ninput), ["element"])
    dset("index_map/feed_pos_disp_m", np.zeros((ninput, 3)), ["element", "xyz"])
    dset(
        "index_map/dish_positions_in_grid_coords",
        rng.uniform(0, 50, (64, 3)),
        ["dish", "xyz"],
    )
    del im

    # Data
    dset("vis", vis, ["frequency", "product", "time"], compress=True)
    dset("vis_weight", weight, ["frequency", "product", "time"], compress=True)
    dset("flags", element_flags, ["frequency", "element", "time"])
    dset(
        "gain",
        np.full((nfreq, ninput, ntime), -1 + 0j, dtype=np.complex64),
        ["frequency", "element", "time"],
    )
    dset(
        "radiometer_chi2",
        np.full((nfreq, ntime, 3), -1, dtype=np.float32),
        ["frequency", "time", "pol_product"],
    )
    dset(
        "eval",
        rng.uniform(size=(nfreq, 2, ntime)).astype(np.float32),
        ["frequency", "eigenval", "time"],
        compress=True,
    )
    dset(
        "evec",
        (rng.normal(size=(nfreq, 2, ninput, ntime)) + 0j).astype(np.complex64),
        ["frequency", "eigenvec", "element", "time"],
        compress=True,
    )
    dset(
        "erms",
        rng.uniform(size=(nfreq, ntime)).astype(np.float32),
        ["frequency", "time"],
    )

    # Data quality
    frame_length = 1949696
    valid = np.round((1 - frac_lost) * frame_length).astype(np.uint64)
    dset("frames_added", frames_added, ["frequency", "time"])
    dset("frac_lost", frac_lost, ["frequency", "time"])
    dset("frac_rfi", frac_lost / 2, ["frequency", "time"])
    dset("frac_rfi_only", frac_lost / 4, ["frequency", "time"])
    dset("frac_pl", frac_lost / 2, ["frequency", "time"])
    dset("valid_fpga_count", valid, ["frequency", "time"])
    dset("rfi_fpga_count", (frame_length - valid) // 2, ["frequency", "time"])
    dset("rfi_only_fpga_count", (frame_length - valid) // 4, ["frequency", "time"])
    dset("pl_fpga_count", (frame_length - valid) // 2, ["frequency", "time"])

    # Timing
    era = np.where(fill, (bin_abs % 8640) / 24.0, 0.0)
    for name, arr in [
        ("time_center_t_inst_ns", t_inst_ns),
        ("time_center_ut1_ns", t_ut1_ns),
        ("bin_t_inst_ns", t_inst_ns - 1000),
        ("bin_ut1_ns", t_ut1_ns - 1000),
        ("bin_abs_index", bin_abs),
        ("bin_ERA_deg", era + 1 / 48),
        ("bin_start_ERA_deg", era),
        ("bin_end_ERA_deg", era + 1 / 24),
        ("bin_start_ERAL_deg", era - 119.6),
        ("bin_end_ERAL_deg", era - 119.6 + 1 / 24),
        ("bin_delta_ut1_inst", np.full(ntime, -0.005)),
        ("bin_xp_as", np.full(ntime, 0.2)),
        ("bin_yp_as", np.full(ntime, 0.33)),
        ("fpga_start_tick", np.arange(ntime, dtype=np.uint64) * frame_length),
        ("frame_length_fpga_ticks", np.full(ntime, frame_length, dtype=np.uint64)),
        ("rfi_frame_excision_enabled", np.zeros(ntime, dtype=bool)),
        ("rfi_frame_excision_num", np.zeros(ntime, dtype=np.int32)),
    ]:
        dset(name, arr, ["time"])
    dset(
        "rfi_frame_excision_threshold",
        np.zeros((ntime, 8), np.float32),
        ["time", "threshold"],
    )
    dset(
        "rfi_frame_excision_fraction",
        np.zeros((ntime, 8), np.float32),
        ["time", "threshold"],
    )

    # A beamformer mask with its own time axis, which readers should ignore
    fh.create_group("bf_mask")
    dset(
        "bf_mask/mask",
        np.ones((7, 2, 2, 64), dtype=np.int8),
        ["time", "stream", "pol", "dish"],
    )

    # Configuration snapshots, the second one (receiver) optionally recording the
    # CRS board remap
    snapshots = [{"config": {"input_order": "CHORDBeamformer", "num_dishes": 64}}]
    if remap is not None:
        snapshots.append(
            {"config": {"dpdk": {"crs_board_remap": list(remap)}, "num_crs_boards": 16}}
        )
    dset(
        "config_json",
        np.array([json.dumps(s) for s in snapshots], dtype=h5py.string_dtype()),
        ["config"],
    )

    # Digital gain archive, with gain_exp = -1 on the used inputs (as for the
    # pathfinder), which readers must ignore
    dg = fh.create_group("digital_gains")
    dg.attrs["acquisition_name"] = "test_digitalgain"
    dg.attrs["selected_update_idx"] = 0
    dg.create_dataset("gain_coeff", data=gain_coeff[np.newaxis])
    gain_exp = np.zeros((1, N_GAIN_INPUT), dtype=np.int32)
    gain_exp[0, _fengine_input(inputs["input_list"], remap or CRS_BOARD_REMAP)] = -1
    dg.create_dataset("gain_exp", data=gain_exp)
    dg.create_dataset("compute_time", data=np.full((1, N_GAIN_INPUT), 1.7e9))
    dg.create_dataset(
        "update_id",
        data=np.array([f"digitalgain_update{update}"], dtype=h5py.string_dtype()),
    )
    dg.create_group("index_map")
    dg.create_dataset("index_map/freq", data=gain_freq)
    gin = np.empty(
        N_GAIN_INPUT, dtype=[("chan_id", "<u2"), ("correlator_input", "S32")]
    )
    gin["chan_id"] = np.arange(N_GAIN_INPUT)
    gin["correlator_input"] = [f"test{i:06d}".encode() for i in range(N_GAIN_INPUT)]
    dg.create_dataset("index_map/input", data=gin)
    dg.create_dataset("index_map/update_time", data=np.array([1.7e9 + update]))
