"""Analysis data format for CHORD.

The CHORD analogue of :mod:`ch_util.andata`. It reads the N\\ :sup:`2`
visibility files written by the CHORD X-engine (kotekan ``hdf5N2Write``, files
``vis_<abs_file_idx>_<YYYYMMDD>T_<HHMMSS>_<ns>.h5``) and concatenates them in
time into a :class:`CorrData` container laid out like CHIME's
:class:`ch_util.andata.CorrData`.

Classes
=======

- :class:`BaseData`
- :class:`CorrData`
- :class:`CorrReader`
- :class:`AnDataError`

Functions
=========

- :func:`andata_from_xengine`
- :func:`concatenate`
- :func:`digital_gain_input`
- :func:`get_crs_board_remap`
- :func:`read_digital_gains`
- :func:`subclass_from_obj`
- :func:`versiontuple`

Analysis format
===============

A :class:`CorrData` built from X-engine files holds:

- ``vis`` : (freq, prod, time) complex64 visibilities.
- ``flags/vis_weight`` : (freq, prod, time) inverse variance weights. Zero where
  there is no data.
- ``flags/inputs`` : (input, time) input flags, 1 for a good input and 0 for a bad
  one, taken from the X-engine's per-element flags.
- ``flags/frac_lost``, ``flags/frac_rfi``, ``flags/frac_rfi_only``,
  ``flags/frac_pl``, ``flags/frames_added``, ``flags/*_fpga_count`` : (freq, time)
  data quality from the X-engine.
- ``flags/element_flags`` : (freq, input, time) the raw X-engine element flags.
- ``digital_gain`` : (freq, input) the F-engine gains that were divided out.
- ``bin_*``, ``time_center_*``, ``fpga_start_tick``, ``frame_length_fpga_ticks``,
  ``rfi_frame_excision_*``, ``file_index`` : per-time information.
- ``input_info/*`` : per-input description (label, type, pol, dish index, grid
  position, feed position).
- ``dish_positions_in_grid_coords``, ``config_json`` : telescope layout and the
  kotekan/FPGA configuration snapshots.
- ``eval``, ``evec``, ``erms`` : the X-engine eigen-decomposition, if present.
- index maps ``freq``, ``prod``, ``input`` (``chan_id``, ``correlator_input``),
  ``time`` (UNIX seconds at the centre of each sample), ``stack`` and the reverse
  map ``stack`` (identities, as the data is unstacked).

The ``input`` index map has ``chan_id`` equal to the element index in the
full-array ``CHORDBeamformer`` ordering (``dish_idx + 64 * pol``), which does not
change when inputs are selected, and ``correlator_input`` equal to the input
label (e.g. ``B4p1``, ``RFIA1p2``).

Differences from the CHIME acquisition format
=============================================

The X-engine files are not read through :func:`concatenate` (as CHIME archive
files are) because:

- there is no ``index_map/time``: the time of each sample is in the
  ``time_center_*`` datasets;
- the axes are called ``frequency``, ``product`` and ``element`` rather than
  ``freq``, ``prod`` and ``input``;
- the weights and data quality datasets are in the root group rather than in
  ``flags``;
- files at the start and end of an acquisition contain unfilled time bins
  (``time_center_t_inst_ns == 0``), which are dropped;
- some axes (e.g. ``radiometer_chi2`` and ``bf_mask``) do not follow the "time is
  the last axis" convention, and ``bf_mask`` has its own time axis.

Digital gains and the CRS board remap
=====================================

The F-engine multiplies the channelised voltage of each input by a complex
digital gain before quantisation, so the X-engine visibilities are
``V_ab = g_a g_b^* V_ab^{true}``. Every X-engine file carries the gain archive
that was in use in ``/digital_gains``, and :class:`CorrData` divides the gains
back out by default (``apply_gain=True``). Two things are needed for this:

**Which gain belongs to which X-engine element.** The gain archive is indexed
by *F-engine input*, ``8 * source_id + lane``: there are 16 CRS boards
(``source_id`` 0-15), each digitising 8 ADC inputs (``lane`` 0-7). The X-engine
elements are in a different order. Its packet capture (kotekan ``dpdkCore``)
places the stream from each board into an *output slot* following the
``dpdk.crs_board_remap`` configuration: ``crs_board_remap[slot]`` is the
``source_id`` of the board feeding that slot. Element ``e`` of the full array is
then lane ``e % 8`` of slot ``e // 8``, and in the ``CHORDBeamformer`` ordering
slots 0-7 hold polarisation 0 and slots 8-15 polarisation 1 of dish groups 0-7.
So the gain of element ``e`` is F-engine input::

    8 * crs_board_remap[e // 8] + e % 8

For the pathfinder wiring (from Sep 2 2026) ``crs_board_remap`` is
``[0, 2, 4, 6, 8, 10, 12, 15, 1, 3, 5, 7, 9, 11, 13, 14]``: the dish boards send
polarisation 0 from even and polarisation 1 from odd ``source_id``\\ s, while the
RFI antenna boards are cabled the other way round (``source_id`` 15 feeds slot 7
and 14 feeds slot 15). Simply using the element index as the F-engine input is
only right for elements 0-7. The mapping is confirmed by the data: the saturated
(~1e10) gains fall exactly on the inputs that carry no signal (A4, A3 and A1 in
both polarisations, and RFIA4p1).

The remap is read from the ``dpdk`` section of the receiver configuration
snapshot in ``/config_json`` when a file records it (see
:func:`get_crs_board_remap`); otherwise :data:`CRS_BOARD_REMAP` is used. It must
be updated if the F-engine boards are re-cabled.

**The size of the gains.** The applied gain is ``gain_coeff``. The archive also
has a per-input exponent ``gain_exp`` (``gain_coeff * 2**gain_exp``), but for
CHORD the F-engine fixes it to -1 and ignores it (it is there for CHIME), so it
is not used. The coefficients are ~1e8-1e10, so ``|g|^4`` would overflow the
float32 weights. The gains are therefore divided by a single number, the median
``|g|`` of the dish inputs over all frequencies of the first file (saved in the
``digital_gain_norm`` attribute), before being applied. This only sets the
overall scale of the data, which calibration fixes later; it does not change
the relative gains between inputs or frequencies, or the signal to noise.

Notes
-----
Parts of this module (the structure of the classes, :class:`CorrReader` and the
selection helpers) are adapted from :mod:`ch_util.andata`, Copyright 2020 CHIME
collaboration, MIT License.

Examples
--------
Read a whole acquisition, keeping only the dish inputs and 100 channels:

>>> from chord_util import andata
>>> reader = andata.CorrReader("/path/to/acq_20260911_232919_986789057/vis_*.h5")
>>> reader.select_input_type("dish")
>>> reader.freq_sel = slice(3000, 3100)
>>> data = reader.read()
>>> print(data.vis.shape)
(100, 528, 4645)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import posixpath

import h5py
import numpy as np

# The large datasets in the X-engine files are compressed with the bitshuffle
# filter. `hdf5plugin` refuses to load once h5py has been imported (which caput
# does), so register the filter via the bitshuffle package.
from bitshuffle import h5  # noqa: F401
from caput import memdata
from caput.astro import time as ctime
from caput.containers import tod
from caput.util import typeutils

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ANDATA_VERSION = "1.0.0"

# Axes over which datasets can be concatenated
CONCATENATION_AXES = ("time",)

# Input types used by the kotekan telescope description (``/index_map/type``)
INPUT_TYPE_FAKE = -1
INPUT_TYPE_DISH = 0
INPUT_TYPE_RFI = 1

#: Source ID of the CRS board feeding each X-engine output slot, for the
#: pathfinder wiring from Sep 2 2026 (kotekan ``config/chord_pathfinder.j2``,
#: ``dpdk.crs_board_remap``). Used when a file does not record it. See the
#: module documentation.
CRS_BOARD_REMAP = (0, 2, 4, 6, 8, 10, 12, 15, 1, 3, 5, 7, 9, 11, 13, 14)

# Number of ADC inputs (lanes) on each CRS board
INPUTS_PER_BOARD = 8

# (freq, time) data quality datasets, stored under `flags/`
FREQ_TIME_FLAGS = (
    "frac_lost",
    "frac_rfi",
    "frac_rfi_only",
    "frac_pl",
    "frames_added",
    "valid_fpga_count",
    "rfi_fpga_count",
    "rfi_only_fpga_count",
    "pl_fpga_count",
)

# Per-time datasets, stored in the root group
TIME_DATASETS = (
    "bin_ERA_deg",
    "bin_abs_index",
    "bin_delta_ut1_inst",
    "bin_start_ERA_deg",
    "bin_end_ERA_deg",
    "bin_start_ERAL_deg",
    "bin_end_ERAL_deg",
    "bin_t_inst_ns",
    "bin_ut1_ns",
    "bin_xp_as",
    "bin_yp_as",
    "time_center_t_inst_ns",
    "time_center_ut1_ns",
    "fpga_start_tick",
    "frame_length_fpga_ticks",
    "rfi_frame_excision_enabled",
    "rfi_frame_excision_num",
    "rfi_frame_excision_threshold",
    "rfi_frame_excision_fraction",
)

# Per-element tables under the raw `/index_map`, stored under `input_info/`
INPUT_TABLES = (
    "label",
    "type",
    "pol",
    "dish_idx",
    "grid_x_idx",
    "grid_y_idx",
    "coelev_disp_deg",
    "feed_pos_disp_m",
)

# Per-element root attributes of the raw files, stored under `input_info/`
INPUT_ATTRS = ("feed_positions_m", "main_array_grid_indices")

#: Datasets with a frequency axis that can be read, as
#: ``{analysis name: (raw name, axes)}``.
CORR_DATASETS = {
    "vis": ("vis", ("freq", "prod", "time")),
    "flags/vis_weight": ("vis_weight", ("freq", "prod", "time")),
    "flags/element_flags": ("flags", ("freq", "input", "time")),
    **{f"flags/{name}": (name, ("freq", "time")) for name in FREQ_TIME_FLAGS},
    "eval": ("eval", ("freq", "ev", "time")),
    "evec": ("evec", ("freq", "ev", "input", "time")),
    "erms": ("erms", ("freq", "time")),
    "gain": ("gain", ("freq", "input", "time")),
    "radiometer_chi2": ("radiometer_chi2", ("freq", "time", "pol_product")),
}

#: Datasets derived when reading.
DERIVED_DATASETS = ("flags/inputs", "digital_gain")

#: Datasets that only hold sentinel values in the pathfinder pipeline (no stage
#: populates them), so are not read unless explicitly requested.
PLACEHOLDER_DATASETS = ("gain", "radiometer_chi2")

# Aliases from the raw dataset names to the analysis names
_RAW_ALIASES = {raw: name for name, (raw, _) in CORR_DATASETS.items()}


# Main Class Definition
# ---------------------


class BaseData(tod.TOData):
    """CHORD data in analysis format.

    Inherits from :class:`caput.containers.tod.TOData`.

    This is intended to be the main data class for the analysis of CHORD data.
    It is laid out very similarly to how the data is stored in analysis format
    hdf5 files, and the data can optionally be stored in such an hdf5 file
    instead of in memory. Datasets may be in the root group, the ``flags``
    group and the ``input_info`` group.

    Parameters
    ----------
    h5_data : h5py.Group, memdata.MemGroup or hdf5 filename, optional
        Underlying h5py like data container where data will be stored. If not
        provided a new :class:`caput.memdata.MemGroup` instance will be created.
    """

    time_axes = CONCATENATION_AXES

    # Convert strings to/from unicode on load and save
    convert_attribute_strings = True
    convert_dataset_strings = True

    # Groups (other than the root) that may contain datasets
    _allowed_groups = ("flags", "input_info")

    def __new__(cls, h5_data=None, **kwargs):
        """Pick which subclass to instantiate based on attributes in data."""
        new_cls = subclass_from_obj(cls, h5_data)

        return super().__new__(new_cls)

    def __init__(self, h5_data=None, **kwargs):
        super().__init__(h5_data, **kwargs)
        if self._data.file.mode == "r+":
            for group in self._allowed_groups:
                self._data.require_group(group)
            self._data.require_group("reverse_map")
            self.attrs["andata_version"] = ANDATA_VERSION

    # - The main interface - #

    @property
    def datasets(self):
        """Stores hdf5 datasets holding all data.

        Do not try to add a new dataset by assigning to an item of this
        property. Use `create_dataset` instead.

        Returns
        -------
        datasets : read only dictionary
            Entries are :mod:`h5py` or :mod:`caput.memdata` datasets in the root
            group.
        """
        return self._group_datasets("/")

    @property
    def flags(self):
        """Datasets representing flags and data weights.

        Returns
        -------
        flags : read only dictionary
            Entries are :mod:`h5py` or :mod:`caput.memdata` datasets in the
            ``flags`` group.
        """
        return self._group_datasets("flags")

    @property
    def input_info(self):
        """Datasets describing each input.

        Returns
        -------
        input_info : read only dictionary
            Entries are :mod:`h5py` or :mod:`caput.memdata` datasets in the
            ``input_info`` group, each with a leading ``input`` axis.
        """
        return self._group_datasets("input_info")

    @property
    def cal(self):
        """Stores calibration schemes for the datasets.

        Each entry is a calibration scheme which itself is a dict storing
        meta-data about calibration.

        Do not try to add a new entry by assigning to an element of this
        property. Use :meth:`~BaseData.create_cal` instead.

        Returns
        -------
        cal : read only dictionary
            Calibration schemes.
        """
        if "cal" not in self._data:
            return memdata.ro_dict({})
        return memdata.ro_dict(
            {name: value.attrs for name, value in self._data["cal"].items()}
        )

    def _group_datasets(self, group):
        """Get a read only dictionary of the datasets in a group."""
        try:
            g = self._data if group == "/" else self._data[group]
        except KeyError:
            return memdata.ro_dict({})

        out = {}
        for name, value in g.items():
            if not memdata.is_group(value):
                out[name] = value
        return memdata.ro_dict(out)

    # - Methods used by base class to control container structure. - #

    def dataset_name_allowed(self, name):
        """Permits datasets in the root, 'flags' and 'input_info' groups."""
        parent_name, _ = posixpath.split(name)
        return parent_name == "/" or self.group_name_allowed(parent_name)

    def group_name_allowed(self, name):
        """Permits only the 'flags' and 'input_info' groups."""
        return name.strip("/") in self._allowed_groups

    # - Methods for manipulating and building the class. - #

    def create_cal(self, name, cal=None):
        """Create a new cal entry.

        Parameters
        ----------
        name : str
            Name of the calibration scheme.
        cal : dict, optional
            Meta-data describing the scheme, stored as attributes.
        """
        if cal is None:
            cal = {}
        self._data.require_group("cal").create_group(name)
        for key, value in cal.items():
            self._data["cal"][name].attrs[key] = value

    def create_flag(self, name, *args, **kwargs):
        """Create a new flags dataset.

        Parameters
        ----------
        name : str
            Name of the dataset in the ``flags`` group.
        *args, **kwargs
            Passed to :meth:`create_dataset`.

        Returns
        -------
        dset : memdata dataset
            The new dataset.
        """
        return self.create_dataset("flags/" + name, *args, **kwargs)

    def create_reverse_map(self, axis_name, reverse_map):
        """Create a new reverse map.

        Parameters
        ----------
        axis_name : str
            Name of the axis the reverse map is for (e.g. ``stack``).
        reverse_map : np.ndarray
            The reverse map.
        """
        return self._data["reverse_map"].create_dataset(axis_name, data=reverse_map)

    def del_reverse_map(self, axis_name):
        """Delete a reverse map.

        Parameters
        ----------
        axis_name : str
            Name of the reverse map to delete.
        """
        del self._data["reverse_map"][axis_name]

    # - These describe the various data axes. - #

    @property
    def ntime(self):
        """Length of the time axis of the visibilities."""
        return len(self.index_map["time"])

    @property
    def time(self):
        """The 'time' axis centres as Unix/POSIX time.

        For CHORD data ``index_map/time`` already holds the UNIX time of the
        centre of the data in each sample.

        Returns
        -------
        time : np.ndarray[ntime]
            UNIX time in seconds.
        """
        return self.index_map["time"][:]

    @classmethod
    def _interpret_and_read(cls, acq_files, start, stop, datasets, out_group):
        """Concatenate files that are already in analysis format.

        Parameters
        ----------
        acq_files : list of h5py.File
            Open files, in time order.
        start, stop : int
            Range of time samples to read over all the files together.
        datasets : list of str
            Names of the datasets to read. `None` reads all of them.
        out_group : h5py.Group or memdata.MemGroup
            Container to store the data in.

        Returns
        -------
        data : BaseData
            The concatenated data.
        """
        andata_objs = [cls(d) for d in acq_files]
        return concatenate(
            andata_objs,
            out_group=out_group,
            start=start,
            stop=stop,
            datasets=datasets,
            convert_attribute_strings=cls.convert_attribute_strings,
            convert_dataset_strings=cls.convert_dataset_strings,
        )

    @classmethod
    def from_acq_h5(
        cls, acq_files, start=None, stop=None, datasets=None, out_group=None, **kwargs
    ):
        """Concatenate files already in analysis format.

        This reads files that were written from a :class:`BaseData` (or subclass)
        container, and concatenates them in time. To read the raw X-engine files
        use :meth:`CorrData.from_acq_h5`.

        Parameters
        ----------
        acq_files : filename, `h5py.File` or list there-of or filename pattern
            Files to concatenate. Filename patterns with wild cards (e.g.
            "foo*.h5") are supported.
        start : integer, optional
            What frame to start at in the full set of files.
        stop : integer, optional
            What frame to stop at in the full set of files.
        datasets : list of strings
            Names of datasets to include. Default is to include all datasets
            found in the files.
        out_group : `h5py.Group`, hdf5 filename or `memdata.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.
        **kwargs
            Passed to `_interpret_and_read` of the class.

        Returns
        -------
        data : BaseData
            Loaded data object.
        """
        acq_files = tod.ensure_file_list(acq_files)
        if not acq_files:
            raise ValueError("Acquisition file list is empty.")

        to_close = [False] * len(acq_files)
        try:
            _open_files(acq_files, to_close)
            data = cls._interpret_and_read(
                acq_files=acq_files,
                start=start,
                stop=stop,
                datasets=datasets,
                out_group=out_group,
                **kwargs,
            )
        finally:
            for ii in range(len(acq_files)):
                if len(to_close) > ii and to_close[ii]:
                    acq_files[ii].close()

        return data

    @property
    def timestamp(self):
        """Deprecated name for :attr:`~BaseData.time`."""
        return self.time

    @staticmethod
    def convert_time(time):
        """Convert a time to UNIX time.

        Parameters
        ----------
        time : float, datetime or skyfield time
            The time to convert.

        Returns
        -------
        unix_time : float
            UNIX time in seconds.
        """
        return ctime.ensure_unix(time)


class CorrData(BaseData):
    """Subclass of :class:`BaseData` for CHORD X-engine correlation data."""

    @property
    def vis(self):
        """Convenience access to the visibilities array.

        Equivalent to `self.datasets['vis']`.
        """
        return self.datasets["vis"]

    @property
    def gain(self):
        """Convenience access to the X-engine gain dataset.

        Equivalent to `self.datasets['gain']`. In the pathfinder pipeline no stage
        applies gains in the X-engine, so this only holds the sentinel -1+0j and is
        not read by default. The F-engine gains are in :attr:`digital_gain`.
        """
        return self.datasets["gain"]

    @property
    def digital_gain(self):
        """Convenience access to the F-engine digital gains.

        Equivalent to `self.datasets['digital_gain']`: the (normalised) gains of
        shape (freq, input) that were divided out of the data if
        ``attrs['digital_gains_applied']`` is True.
        """
        return self.datasets["digital_gain"]

    @property
    def weight(self):
        """Convenience access to the visibility weight array.

        Equivalent to `self.flags['vis_weight']`.
        """
        return self.flags["vis_weight"]

    @property
    def input_flags(self):
        """Convenience access to the input flags dataset.

        Equivalent to `self.flags['inputs']`.
        """
        return self.flags["inputs"]

    @property
    def frac_lost(self):
        """Convenience access to the fraction of each sample that was lost.

        Equivalent to `self.flags['frac_lost']`.
        """
        return self.flags["frac_lost"]

    @property
    def nprod(self):
        """Length of the prod axis."""
        return len(self.index_map["prod"])

    @property
    def prod(self):
        """The correlation product axis as channel pairs."""
        return self.index_map["prod"]

    @property
    def nfreq(self):
        """Length of the freq axis."""
        return len(self.index_map["freq"])

    @property
    def freq(self):
        """The spectral frequency axis as bin centres in MHz."""
        return self.index_map["freq"]["centre"]

    @property
    def ninput(self):
        """Length of the input axis."""
        return len(self.index_map["input"])

    @property
    def input(self):
        """The inputs as (`chan_id`, `correlator_input`) pairs."""
        return self.index_map["input"]

    @property
    def nstack(self):
        """Length of the stack axis."""
        return len(self.index_map["stack"])

    @property
    def stack(self):
        """The stack axis, as (`prod`, `conjugate`) pairs."""
        return self.index_map["stack"]

    @property
    def prodstack(self):
        """A pair of input indices representative of those in the stack.

        Note, these are correctly conjugated on return, and so calculations
        of the baseline and polarisation can be done without additionally
        looking up the stack conjugation.
        """
        if not self.is_stacked:
            return self.prod

        t = self.index_map["prod"][:][self.index_map["stack"]["prod"]]

        prodmap = t.copy()
        conj = self.stack["conjugate"]
        prodmap["input_a"] = np.where(conj, t["input_b"], t["input_a"])
        prodmap["input_b"] = np.where(conj, t["input_a"], t["input_b"])

        return prodmap

    @property
    def is_stacked(self):
        """Is the data stacked? X-engine data is always unstacked."""
        return "stack" in self.index_map and len(self.stack) != len(self.prod)

    @property
    def labels(self):
        """The label of each input, e.g. `B4p1` or `RFIA1p2`."""
        return np.array(
            [typeutils.bytes_to_unicode(x) for x in self.input["correlator_input"]]
        )

    @property
    def input_type(self):
        """The type of each input (-1 fake, 0 dish, 1 RFI monitor)."""
        return self.input_info["type"][:]

    @property
    def is_dish(self):
        """Boolean mask selecting the dish inputs."""
        return self.input_type == INPUT_TYPE_DISH

    @property
    def is_rfi_monitor(self):
        """Boolean mask selecting the RFI monitor inputs."""
        return self.input_type == INPUT_TYPE_RFI

    @classmethod
    def _interpret_and_read(
        cls,
        acq_files,
        start,
        stop,
        datasets,
        out_group,
        stack_sel=None,
        prod_sel=None,
        input_sel=None,
        freq_sel=None,
        apply_gain=True,
        normalise_gain=True,
        crs_board_remap=None,
        file_info=None,
    ):
        """Read X-engine files.

        See :meth:`from_acq_h5` for the parameters. `file_info` is the output of
        :func:`_scan_xengine_files`, if it has already been computed.
        """
        if file_info is None:
            file_info = _scan_xengine_files(acq_files, normalise_gain, crs_board_remap)

        return andata_from_xengine(
            cls,
            file_info,
            start,
            stop,
            stack_sel,
            prod_sel,
            input_sel,
            freq_sel,
            datasets,
            out_group,
            apply_gain,
        )

    @classmethod
    def from_acq_h5(cls, acq_files, start=None, stop=None, **kwargs):
        """Convert X-engine format hdf5 data to analysis data object.

        This method overloads the one in BaseData.

        Reads hdf5 data produced by the X-engine and converts it to analysis
        format in memory.

        Parameters
        ----------
        acq_files : filename, `h5py.File` or list there-of or filename pattern
            Files to convert from X-engine format to analysis format. Filename
            patterns with wild cards (e.g. "foo*.h5") are supported. The files are
            sorted by their absolute file index.
        start : integer, optional
            What frame to start at in the full set of files. Frames count the
            filled time samples of all the files together.
        stop : integer, optional
            What frame to stop at in the full set of files.
        stack_sel : valid numpy index
            Used to select a subset of the stacked correlation products. As the
            X-engine data is unstacked this is the same as *prod_sel*.
            Only one of *stack_sel*, *prod_sel*, and *input_sel* may be
            specified.
        prod_sel : valid numpy index
            Used to select a subset of correlation products. The inputs involved
            in the selected products are kept.
            Only one of *stack_sel*, *prod_sel*, and *input_sel* may be
            specified.
        input_sel : valid numpy index
            Used to select a subset of correlator inputs. The products between
            the selected inputs are kept.
            Only one of *stack_sel*, *prod_sel*, and *input_sel* may be
            specified.
        freq_sel : valid numpy index
            Used to select a subset of frequencies (indices into the frequency
            axis of the files).
        datasets : list of strings
            Names of datasets to include (analysis format names, e.g.
            ``"vis"``, ``"flags/vis_weight"``; the raw names such as
            ``"vis_weight"`` are also accepted). The per-time, per-input and
            telescope description datasets are always read. Requested datasets
            not in the files are skipped. Default is all of
            :data:`CORR_DATASETS` and :data:`DERIVED_DATASETS` found in the files,
            except :data:`PLACEHOLDER_DATASETS`.
        out_group : `h5py.Group`, hdf5 filename or `memdata.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance. Not supported with *distributed*.
        apply_gain : boolean, optional
            Whether to divide the F-engine digital gains out of the visibilities
            (and scale the weights to match). Default is True.
        normalise_gain : boolean, optional
            Whether to normalise the gains by the median dish gain before applying
            them. Default is True. See the module documentation.
        crs_board_remap : list of int, optional
            Source ID of the CRS board feeding each X-engine slot, used to find
            the digital gain of each input. Default is to read it from the files
            (falling back to :data:`CRS_BOARD_REMAP`).
        renormalize : boolean, optional
            Accepted for compatibility with :mod:`ch_util.andata` and ignored: the
            X-engine already averages over the valid FPGA samples.
        distributed : boolean, optional
            Load data into distributed datasets, split over frequency.
        comm : MPI.Comm
            Communicator to distributed over. Use MPI.COMM_WORLD if not set.

        Returns
        -------
        data : CorrData
            Loaded data object.

        Raises
        ------
        ValueError
            If no files are found, unknown keyword arguments are given, or the
            selections are inconsistent.
        RuntimeError
            If the files have different layouts or their times overlap.

        Examples
        --------
        Read a set of files into one timestream:

        >>> data = CorrData.from_acq_h5("acq_20260911_232919_986789057/vis_*.h5")

        Only a range of frequencies and time samples:

        >>> data = CorrData.from_acq_h5(files, start=10, stop=370,
        ...                             freq_sel=slice(3000, 3100))

        Only the dish inputs:

        >>> reader = CorrReader(files)
        >>> dishes = np.flatnonzero(reader.input_type == INPUT_TYPE_DISH)
        >>> data = CorrData.from_acq_h5(files, input_sel=dishes)
        """
        stack_sel = kwargs.pop("stack_sel", None)
        prod_sel = kwargs.pop("prod_sel", None)
        input_sel = kwargs.pop("input_sel", None)
        freq_sel = kwargs.pop("freq_sel", None)
        datasets = kwargs.pop("datasets", None)
        out_group = kwargs.pop("out_group", None)
        apply_gain = kwargs.pop("apply_gain", True)
        normalise_gain = kwargs.pop("normalise_gain", True)
        crs_board_remap = kwargs.pop("crs_board_remap", None)
        kwargs.pop("renormalize", None)
        distributed = kwargs.pop("distributed", False)
        comm = kwargs.pop("comm", None)
        file_info = kwargs.pop("_file_info", None)

        if kwargs:
            raise ValueError(f"Received unknown keyword arguments {kwargs.keys()}.")

        files = _ensure_file_names(acq_files)

        # If want a distributed file, just pass straight off to a private method
        if distributed:
            return cls._from_acq_h5_distributed(
                acq_files=files,
                start=start,
                stop=stop,
                datasets=datasets,
                stack_sel=stack_sel,
                prod_sel=prod_sel,
                input_sel=input_sel,
                freq_sel=freq_sel,
                apply_gain=apply_gain,
                normalise_gain=normalise_gain,
                crs_board_remap=crs_board_remap,
                comm=comm,
                file_info=file_info,
            )

        return cls._interpret_and_read(
            acq_files=files,
            start=start,
            stop=stop,
            datasets=datasets,
            out_group=out_group,
            stack_sel=stack_sel,
            prod_sel=prod_sel,
            input_sel=input_sel,
            freq_sel=freq_sel,
            apply_gain=apply_gain,
            normalise_gain=normalise_gain,
            crs_board_remap=crs_board_remap,
            file_info=file_info,
        )

    @classmethod
    def _from_acq_h5_distributed(
        cls,
        acq_files,
        start,
        stop,
        stack_sel,
        prod_sel,
        input_sel,
        freq_sel,
        datasets,
        apply_gain,
        normalise_gain,
        crs_board_remap,
        comm,
        file_info=None,
    ):
        """Read X-engine files into a container distributed over frequency.

        Rank 0 reads the file metadata and shares it. Each rank then reads only
        its own frequencies from every file, and the local arrays are wrapped into
        distributed datasets.
        """
        from caput import mpiarray
        from caput.util import mpitools
        from mpi4py import MPI

        if comm is None:
            comm = MPI.COMM_WORLD

        # Scan the files once, on rank 0
        if file_info is None:
            file_info = comm.bcast(
                (
                    _scan_xengine_files(acq_files, normalise_gain, crs_board_remap)
                    if comm.rank == 0
                    else None
                ),
                root=0,
            )

        # Calculate the global frequency selection, then the local one
        nfreq = len(file_info["freq"])
        freq_ind = np.arange(nfreq)[_ensure_1D_selection(freq_sel)]
        _, f_start, f_end = mpitools.split_local(len(freq_ind), comm=comm)
        local_freq_sel = freq_ind[f_start:f_end]

        # Load just the local part of the data.
        local_data = cls._interpret_and_read(
            acq_files=acq_files,
            start=start,
            stop=stop,
            datasets=datasets,
            out_group=None,
            stack_sel=stack_sel,
            prod_sel=prod_sel,
            input_sel=input_sel,
            freq_sel=local_freq_sel,
            apply_gain=apply_gain,
            file_info=file_info,
        )

        # Initialise distributed container
        data = cls(distributed=True, comm=comm)

        # Copy over the attributes
        memdata.copyattrs(
            local_data.attrs, data.attrs, convert_strings=cls.convert_attribute_strings
        )
        data.attrs["num_file_f"] = len(freq_ind)

        # Iterate over the datasets (in all groups) and copy them over. Those
        # with a leading frequency axis are distributed.
        for name in _walk_datasets(local_data._data):
            old_dset = local_data[name]
            axes = typeutils.bytes_to_unicode(old_dset.attrs["axis"])

            if axes[0] == "freq":
                array = mpiarray.MPIArray.wrap(old_dset[:], axis=0, comm=comm)
            else:
                array = old_dset[:]

            new_dset = data.create_dataset(name, data=array)
            memdata.copyattrs(
                old_dset.attrs,
                new_dset.attrs,
                convert_strings=cls.convert_attribute_strings,
            )

        # Copy over index maps
        for name, index_map in local_data.index_map.items():
            index_map = index_map[:]

            # We need to explicitly stitch the frequency map back together
            if name == "freq":
                index_map = np.concatenate(comm.allgather(index_map))

            data.create_index_map(name, index_map)
            memdata.copyattrs(local_data.index_attrs[name], data.index_attrs[name])

        # Copy over reverse maps
        for name, reverse_map in local_data.reverse_map.items():
            data.create_reverse_map(name, reverse_map[:])

        # An input is good if the X-engine flagged it good at any frequency on any
        # rank
        if "flags/inputs" in data:
            inputs = np.ascontiguousarray(data["flags/inputs"][:], dtype=np.float32)
            comm.Allreduce(MPI.IN_PLACE, inputs, op=MPI.MAX)
            data["flags/inputs"][:] = inputs

        return data

    @classmethod
    def from_acq_h5_fast(cls, fname, comm=None, freq_sel=None, start=None, stop=None):
        """Efficiently read X-engine files in a distributed fashion.

        In contrast to :meth:`from_acq_h5` it is more restrictive, allowing only
        contiguous frequency slices and reading all the (default) datasets.

        Parameters
        ----------
        fname : str or list of str
            File name(s) to read. Unlike CHIME, several files (e.g. a whole
            acquisition) may be given.
        comm : MPI.Comm, optional
            MPI communicator to distribute over. By default this will
            use `MPI.COMM_WORLD`.
        freq_sel : slice, optional
            A selection over the frequency axis. Only `slice` objects
            are supported. If not set, read all frequencies.
        start, stop : int, optional
            Start and stop indexes of the time selection.

        Returns
        -------
        data : CorrData
            The distributed container.
        """
        if freq_sel is None:
            freq_sel = slice(None)
        elif not isinstance(freq_sel, slice):
            raise ValueError(f"freq_sel must be a slice object, not {freq_sel!r}")

        return cls.from_acq_h5(
            fname,
            start=start,
            stop=stop,
            freq_sel=freq_sel,
            distributed=True,
            comm=comm,
        )


# For backwards compatibility with ch_util.
AnData = CorrData


class CorrReader(tod.TODReader):
    """Provides high level reading of CHORD X-engine data.

    Parses and stores meta-data from the file headers allowing for the
    interpretation and selection of the data without reading it all from disk.
    Mirrors :class:`ch_util.andata.CorrReader`.

    Parameters
    ----------
    files : filename, `h5py.File` or list there-of or filename pattern
        Files containing data. Filename patterns with wild cards (e.g.
        "foo*.h5") are supported.
    normalise_gain : bool, optional
        Normalise the digital gains by the median dish gain. Default is True.
    crs_board_remap : list of int, optional
        Source ID of the CRS board feeding each X-engine slot. Default is to read
        it from the files.

    Attributes
    ----------
    apply_gain : bool
        Divide out the digital gains when reading. Default is True.
    renormalize : bool
        Accepted for compatibility with :mod:`ch_util.andata`, has no effect.
    distributed : bool
        Read into distributed datasets. Default is False.
    comm : MPI.Comm
        Communicator for distributed reads. Default is `MPI.COMM_WORLD`.
    """

    data_class = CorrData

    def __init__(self, files, normalise_gain=True, crs_board_remap=None):
        files = _ensure_file_names(files)
        self._file_info = _scan_xengine_files(files, normalise_gain, crs_board_remap)
        info = self._file_info

        # Set the metadata attributes used by `tod.TODReader`
        self._files = tuple(info["files"])
        self._time = info["time"]
        self._datasets = tuple(info["available"])
        self._time_sel = (0, len(self._time))
        self._dataset_sel = tuple(_resolve_datasets(None, info["available"]))

        self._prod = info["prod"]
        self._input = _input_map(info, np.arange(len(info["input_list"])))
        self._freq = info["freq"]

        self._prod_sel = None
        self._input_sel = None
        self._freq_sel = None

        # Options passed to CorrData.from_acq_h5() when the read() method is
        # called.
        self.apply_gain = True
        self.renormalize = True
        self.distributed = False
        self.comm = None

    # Properties
    # ----------

    @property
    def prod(self):
        """Correlation products in data files."""
        return self._prod[:].copy()

    @property
    def input(self):
        """Correlator inputs in data files, as (`chan_id`, `correlator_input`)."""
        return self._input[:].copy()

    @property
    def input_type(self):
        """Type of each input in the data files (-1 fake, 0 dish, 1 RFI monitor)."""
        return self._file_info["input_tables"]["type"].copy()

    @property
    def freq(self):
        """Spectral frequency bin centres in data files."""
        return self._freq[:].copy()

    @property
    def prod_sel(self):
        """Which correlation products to read.

        Returns
        -------
        prod_sel : 1D data selection
            Valid numpy index for a 1D array, specifying what data to read
            along the correlation product axis.
        """
        return self._prod_sel

    @prod_sel.setter
    def prod_sel(self, value):
        if value is not None:
            # Check to make sure this is a valid index for the product axis.
            self.prod["input_a"][value]
            if self.input_sel is not None:
                raise ValueError(
                    "*input_sel* is set and cannot specify both *prod_sel*"
                    " and *input_sel*."
                )
        self._prod_sel = value

    @property
    def input_sel(self):
        """Which correlator inputs to read.

        Returns
        -------
        input_sel : 1D data selection
            Valid numpy index for a 1D array, specifying what data to read
            along the input axis.
        """
        return self._input_sel

    @input_sel.setter
    def input_sel(self, value):
        if value is not None:
            # Check to make sure this is a valid index for the input axis.
            self.input["chan_id"][value]
            if self.prod_sel is not None:
                raise ValueError(
                    "*prod_sel* is set and cannot specify both *prod_sel*"
                    " and *input_sel*."
                )
        self._input_sel = value

    @property
    def freq_sel(self):
        """Which frequencies to read.

        Returns
        -------
        freq_sel : 1D data selection
            Valid numpy index for a 1D array, specifying what data to read
            along the frequency axis.
        """
        return self._freq_sel

    @freq_sel.setter
    def freq_sel(self, value):
        if value is not None:
            # Check to make sure this is a valid index for the frequency axis.
            self.freq["centre"][value]
        self._freq_sel = value

    # Data Selection Methods
    # ----------------------

    def select_prod_pairs(self, pairs):
        """Sets :attr:`~CorrReader.prod_sel` to include given product pairs.

        Parameters
        ----------
        pairs : list of integer pairs
            Input pairs to be included.
        """
        self.input_sel = None
        sel = []
        for input_a, input_b in pairs:
            for ii in range(len(self.prod)):
                p_input_a, p_input_b = self.prod[ii]
                if (input_a == p_input_a and input_b == p_input_b) or (
                    input_a == p_input_b and input_b == p_input_a
                ):
                    sel.append(ii)
        self.prod_sel = sorted(sel)

    def select_prod_autos(self):
        """Sets :attr:`~CorrReader.prod_sel` to only auto-correlations."""
        self.input_sel = None
        sel = []
        for ii, prod in enumerate(self.prod):
            if prod[0] == prod[1]:
                sel.append(ii)
        self.prod_sel = sel

    def select_prod_by_input(self, input):
        """Sets :attr:`~CorrReader.prod_sel` to only products with given input.

        Parameters
        ----------
        input : integer
            Correlator input number (index into the input axis). All correlation
            products with this input as one of the pairs are selected.
        """
        self.input_sel = None
        sel = []
        for ii, prod in enumerate(self.prod):
            if prod[0] == input or prod[1] == input:
                sel.append(ii)
        self.prod_sel = sel

    def select_input_type(self, input_type):
        """Sets :attr:`~CorrReader.input_sel` to the inputs of a given type.

        Parameters
        ----------
        input_type : {"dish", "rfi", "all"}
            The dish inputs, the RFI monitor inputs, or all inputs.
        """
        types = {"dish": [INPUT_TYPE_DISH], "rfi": [INPUT_TYPE_RFI], "all": None}
        if input_type not in types:
            raise ValueError(f"input_type must be one of {list(types)}.")

        self.prod_sel = None
        if types[input_type] is None:
            self.input_sel = None
        else:
            self.input_sel = np.flatnonzero(np.isin(self.input_type, types[input_type]))

    def select_input_labels(self, labels):
        """Sets :attr:`~CorrReader.input_sel` to the inputs with given labels.

        Parameters
        ----------
        labels : list of str
            Input labels, e.g. ``["A2p1", "A2p2"]``.
        """
        all_labels = [
            typeutils.bytes_to_unicode(x) for x in self.input["correlator_input"]
        ]
        missing = set(labels) - set(all_labels)
        if missing:
            raise ValueError(f"Unknown input labels {sorted(missing)}.")

        self.prod_sel = None
        self.input_sel = np.array(sorted(all_labels.index(lbl) for lbl in labels))

    def select_freq_range(self, freq_low=None, freq_high=None, freq_step=None):
        """Sets :attr:`~CorrReader.freq_sel` to given physical frequency range.

        Frequencies selected will have bin centres bracketed by provided range.

        Parameters
        ----------
        freq_low : float
            Lower end of the frequency range in MHz.  Default is the lower edge
            of the band.
        freq_high : float
            Upper end of the frequency range in MHz.  Default is the upper edge
            of the band.
        freq_step : float
            How much bandwidth to skip over between samples in MHz. This value
            is approximate. Default is to include all samples in given range.
        """
        freq = self.freq["centre"]
        nfreq = len(freq)
        if freq_step is None:
            step = 1
        else:
            df = abs(np.mean(np.diff(freq)))
            step = max(int(freq_step // df), 1)
        # Unlike CHIME, frequencies are in increasing order.
        start = 0 if freq_low is None else int(np.searchsorted(freq, freq_low))
        stop = nfreq if freq_high is None else int(np.searchsorted(freq, freq_high, "right"))
        self.freq_sel = np.s_[start:stop:step]

    def select_freq_physical(self, frequencies):
        """Sets :attr:`~CorrReader.freq_sel` to include given physical frequencies.

        Parameters
        ----------
        frequencies : list of floats
            Frequencies to select. Physical frequencies are matched to indices
            on a best match basis.

        Raises
        ------
        ValueError
            If a frequency is not within any channel.
        """
        freq_centre = self.freq["centre"]
        freq_width = self.freq["width"]
        frequencies = np.array(frequencies)
        n_sel = len(frequencies)
        diff_freq = abs(freq_centre - frequencies[:, None])
        match_mask = diff_freq < freq_width / 2
        freq_inds = []
        for ii in range(n_sel):
            matches = np.where(match_mask[ii, :])
            try:
                first_match = matches[0][0]
            except IndexError:
                raise ValueError(f"No match for frequency {frequencies[ii]} MHz.")
            freq_inds.append(first_match)
        self.freq_sel = sorted(set(freq_inds))

    # Data Reading
    # ------------

    def read(self, out_group=None):
        """Read the selected data.

        Parameters
        ----------
        out_group : `h5py.Group`, hdf5 filename or `memdata.Group`
            Underlying hdf5 like container that will store the data for the
            BaseData instance.

        Returns
        -------
        data : :class:`CorrData`
            Data read from :attr:`~CorrReader.files` based on the selections given
            in :attr:`~CorrReader.time_sel`, :attr:`~CorrReader.prod_sel`,
            :attr:`~CorrReader.input_sel`, and :attr:`~CorrReader.freq_sel`.
        """
        return CorrData.from_acq_h5(
            self.files,
            start=self.time_sel[0],
            stop=self.time_sel[1],
            prod_sel=self.prod_sel,
            freq_sel=self.freq_sel,
            input_sel=self.input_sel,
            apply_gain=self.apply_gain,
            distributed=self.distributed,
            comm=self.comm,
            datasets=self.dataset_sel,
            out_group=out_group,
            _file_info=self._file_info,
        )


# For backwards compatibility with ch_util.
Reader = CorrReader


class AnDataError(Exception):
    """Exception raised when something unexpected happens with the data."""

    pass


# Functions
# ---------

# In caput now.
concatenate = tod.concatenate


def subclass_from_obj(cls, obj):
    """Pick a subclass of :class:`BaseData` based on an input object.

    Parameters
    ----------
    cls : subclass of :class:`BaseData` (class, not an instance)
        Default class to return.
    obj : :class:`h5py.Group`, filename, :class:`memdata.Group` or
          :class:`BaseData` object from which to determine the appropriate
          subclass of :class:`BaseData`.

    Returns
    -------
    new_cls : subclass of :class:`BaseData`
        :class:`CorrData` for correlator data (analysis format files with
        ``acquisition_type == "corr"``, or raw X-engine files), otherwise `cls`.
    """
    # If obj is a filename, open it and recurse.
    if isinstance(obj, str):
        with h5py.File(obj, "r") as f:
            return subclass_from_obj(cls, f)

    new_cls = cls
    attrs = getattr(obj, "attrs", {})
    acquisition_type = typeutils.bytes_to_unicode(attrs.get("acquisition_type", None))
    file_mode = typeutils.bytes_to_unicode(attrs.get("file_mode", None))

    if acquisition_type == "corr" or file_mode == "CHORD":
        new_cls = CorrData
    elif acquisition_type is None and isinstance(obj, BaseData):
        new_cls = obj.__class__
    return new_cls


def versiontuple(v):
    """Create a version tuple from a version string.

    Parameters
    ----------
    v : str
        A version string, e.g. ``"1.0.0"``. A prefix before an underscore (as in
        the X-engine ``"CHORD_0.0"``) is dropped.

    Returns
    -------
    versiontuple : tuple
        A tuple of `int` values created by splitting the string on dots.
    """
    return tuple(map(int, (v.split("_")[-1].split("."))))


def digital_gain_input(input_list, crs_board_remap=CRS_BOARD_REMAP):
    """Get the F-engine input feeding each X-engine element.

    The digital gain archive is indexed by F-engine input,
    ``8 * source_id + lane``, while the X-engine elements are ordered by output
    slot. Element ``e`` of the full array is lane ``e % 8`` of slot ``e // 8``,
    which is fed by board ``crs_board_remap[e // 8]``. See the module
    documentation.

    Parameters
    ----------
    input_list : array_like of int
        Full-array element index of each element, in the ``CHORDBeamformer``
        ordering (the ``input_list`` attribute of the X-engine files, or
        ``index_map/input['chan_id']`` of a :class:`CorrData`).
    crs_board_remap : list of int
        Source ID of the CRS board feeding each X-engine output slot.

    Returns
    -------
    fengine_input : np.ndarray[ninput]
        Index into the ``input`` axis of ``/digital_gains/gain_coeff`` for each
        element.

    Examples
    --------
    >>> digital_gain_input([0, 7, 8, 64, 71, 60])
    array([  0,   7,  16,   8,  15, 124])
    """
    input_list = np.asarray(input_list)
    remap = np.asarray(crs_board_remap)
    slot, lane = np.divmod(input_list, INPUTS_PER_BOARD)
    return remap[slot] * INPUTS_PER_BOARD + lane


def get_crs_board_remap(fh):
    """Get the CRS board remap recorded in an X-engine file.

    The receiver configuration snapshots in ``/config_json`` record the
    ``dpdk.crs_board_remap`` setting of the X-engine packet capture in some
    acquisitions (the first ones), but not in all of them.

    Parameters
    ----------
    fh : h5py.File
        An open X-engine file.

    Returns
    -------
    crs_board_remap : list of int or None
        Source ID of the CRS board feeding each output slot, or None if the file
        does not record it.

    Raises
    ------
    AnDataError
        If the configuration snapshots in the file disagree with each other.
    """
    if "config_json" not in fh:
        return None

    found = set()
    for snapshot in fh["config_json"][:]:
        try:
            config = json.loads(snapshot).get("config", {})
        except (ValueError, AttributeError):
            continue
        remap = config.get("dpdk", {}).get("crs_board_remap", None)
        if remap is not None:
            found.add(tuple(int(x) for x in remap))

    if len(found) > 1:
        raise AnDataError(f"Inconsistent crs_board_remap in {fh.filename}: {found}")
    return list(found.pop()) if found else None


def read_digital_gains(fh, freq_centre, fengine_input):
    """Read the F-engine digital gains from an X-engine file.

    The gain applied to each input is ``gain_coeff``. The exponent
    ``gain_exp`` in the archive is fixed to -1 and ignored by the CHORD F-engine,
    so it is not used.

    Parameters
    ----------
    fh : h5py.File
        An open X-engine file.
    freq_centre : np.ndarray[nfreq]
        Centres (MHz) of the frequency channels to extract.
    fengine_input : np.ndarray[ninput]
        F-engine input of each element, see :func:`digital_gain_input`.

    Returns
    -------
    gain : np.ndarray[nfreq, ninput]
        The complex gains of the currently selected update
        (``/digital_gains.attrs['selected_update_idx']``).

    Raises
    ------
    AnDataError
        If the frequencies can't be found in the gain archive.
    """
    dg = fh["digital_gains"]
    upd = int(dg.attrs.get("selected_update_idx", 0))
    gain_freq = dg["index_map/freq"]["centre"][:]

    freq_id = np.clip(np.searchsorted(gain_freq, freq_centre), 0, len(gain_freq) - 1)
    if not np.allclose(gain_freq[freq_id], freq_centre):
        raise AnDataError("Could not match frequencies to the digital gain table.")

    if len(freq_id) == 0:
        return np.zeros((0, len(fengine_input)), dtype=np.complex128)

    fstart, fstop = freq_id.min(), freq_id.max() + 1
    coeff = dg["gain_coeff"][upd, fstart:fstop][freq_id - fstart][:, fengine_input]

    return coeff.astype(np.complex128)


def andata_from_xengine(
    cls,
    file_info,
    start,
    stop,
    stack_sel,
    prod_sel,
    input_sel,
    freq_sel,
    datasets,
    out_group,
    apply_gain,
):
    """Create an Andata object from X-engine files.

    The CHORD analogue of ``ch_util.andata.andata_from_archive2``.

    Parameters
    ----------
    cls : class
        Class of object to create.
    file_info : dict
        Metadata of the files, from :func:`_scan_xengine_files`.
    start : int
        What frame to start at in the full set of files.
    stop : int
        What frame to stop at in the full set of files.
    stack_sel, prod_sel, input_sel : 1D data selection
        Valid numpy indices along the stack, product and input axes. Only one may
        be given.
    freq_sel : 1D data selection
        Valid numpy index for a 1D array, specifying what data to read
        along the frequency axis.
    datasets : list of strings
        Names of datasets to include. Default is to include all datasets
        found in the files except the placeholders.
    out_group : `h5py.Group`, hdf5 filename or `memdata.Group`
        Underlying hdf5 like container that will store the data for the
        BaseData instance.
    apply_gain : bool
        Divide out the digital gains.

    Returns
    -------
    andata : `cls` instance
        The andata object for the requested data.
    """
    info = file_info
    ninput_file = len(info["input_list"])
    nprod_file = len(info["prod"])

    # Unstacked, so the stack and prod axes are the same.
    stack_map = np.empty(nprod_file, dtype=[("prod", "<u4"), ("conjugate", "u1")])
    stack_map["conjugate"][:] = 0
    stack_map["prod"] = np.arange(nprod_file)
    stack_rmap = np.empty(nprod_file, dtype=[("stack", "<u4"), ("conjugate", "u1")])
    stack_rmap["conjugate"][:] = 0
    stack_rmap["stack"] = np.arange(nprod_file)
    if stack_sel is not None:
        if prod_sel is not None:
            raise ValueError("Only one of *stack_sel* and *prod_sel* may be given.")
        prod_sel, stack_sel = stack_sel, None

    (
        stack_sel,
        stack_map,
        stack_rmap,
        prod_sel,
        prod_map,
        input_sel,
        input_map,
    ) = _resolve_stack_prod_input_sel(
        stack_sel,
        stack_map,
        stack_rmap,
        prod_sel,
        info["prod"].copy(),
        input_sel,
        _input_map(info, np.arange(ninput_file)),
    )

    freq_ind = np.arange(len(info["freq"]))[_ensure_1D_selection(freq_sel)]
    input_ind = np.arange(ninput_file)[input_sel]
    prod_ind = np.arange(nprod_file)[prod_sel]
    time_sel = _resolve_time_sel(info, start, stop)
    dsets = _resolve_datasets(datasets, info["available"])

    nfreq, ninput, nprod = len(freq_ind), len(input_ind), len(prod_ind)
    ntime = int(sum(s.sum() for s in time_sel))
    axis_len = {
        "freq": nfreq,
        "prod": nprod,
        "input": ninput,
        "time": ntime,
        "ev": info["n_ev"],
        "pol_product": 3,
    }

    # Frequencies are read as a contiguous slab, then down-selected
    f0, f1 = (int(freq_ind.min()), int(freq_ind.max()) + 1) if nfreq else (0, 0)
    fslab = slice(f0, f1)
    fsub = freq_ind - f0
    freq_centre = info["freq"]["centre"][freq_ind]

    # The element flags are always read, as they give the input flags
    read_dsets = [d for d in dsets if d in CORR_DATASETS]
    if "flags/inputs" in dsets and "flags/element_flags" not in read_dsets:
        read_dsets.append("flags/element_flags")

    arrays = {}
    for name in read_dsets:
        axes = CORR_DATASETS[name][1]
        shape = tuple(axis_len[ax] for ax in axes)
        arrays[name] = np.zeros(shape, dtype=info["dtypes"][name])

    gain = np.zeros((nfreq, ninput), dtype=np.complex128)
    gain_key = None
    denom = None

    t0 = 0
    for ii, (fname, tsel) in enumerate(zip(info["files"], time_sel)):
        nt = int(tsel.sum())
        if nt == 0:
            continue
        tsl = slice(t0, t0 + nt)
        t0 += nt
        if nfreq == 0:
            continue

        with h5py.File(fname, "r") as fh:
            # Get the gains to divide out, only re-reading when they change
            if info["has_gains"] and info["gain_keys"][ii] != gain_key:
                gain_key = info["gain_keys"][ii]
                g = read_digital_gains(
                    fh, freq_centre, info["fengine_input"][input_ind]
                )
                g /= info["gain_norm"]
                if gain_key == info["gain_keys"][0]:
                    gain[:] = g
                denom = g[:, prod_map["input_a"]] * g[:, prod_map["input_b"]].conj()

            for name in read_dsets:
                raw, axes = CORR_DATASETS[name]
                arr = fh[raw][fslab][fsub]
                # Select products/inputs, then the filled time bins of this file
                for iax, ax in enumerate(axes[1:], start=1):
                    if ax == "prod":
                        arr = np.take(arr, prod_ind, axis=iax)
                    elif ax == "input":
                        arr = np.take(arr, input_ind, axis=iax)
                arr = np.compress(tsel, arr, axis=axes.index("time"))
                out_sel = tuple(tsl if ax == "time" else slice(None) for ax in axes)
                arrays[name][out_sel] = arr

            # Samples with no data should have zero weight
            if "flags/vis_weight" in arrays:
                fa = np.compress(tsel, fh["frames_added"][fslab][fsub], axis=1)
                fl = np.compress(tsel, fh["frac_lost"][fslab][fsub], axis=1)
                missing = (fa == 0) | (fl >= 1.0)
                arrays["flags/vis_weight"][..., tsl] *= ~missing[:, np.newaxis, :]

        # Divide out the gains, scaling the weights to match
        if apply_gain and denom is not None:
            good = denom != 0
            if "vis" in arrays:
                vis = arrays["vis"][..., tsl]
                vis[good] /= denom[good][:, np.newaxis]
            if "flags/vis_weight" in arrays:
                weight = arrays["flags/vis_weight"][..., tsl]
                weight *= (np.abs(denom) ** 2)[..., np.newaxis]
                weight[~good] = 0.0

    # Create the container
    data = cls(out_group)

    # Mask over the filled time samples (which the scanned arrays hold)
    time_mask = np.concatenate([s[v] for s, v in zip(time_sel, info["valid"])])

    data.create_index_map("freq", info["freq"][freq_ind])
    data.create_index_map("prod", prod_map)
    data.create_index_map("input", input_map)
    data.create_index_map("time", info["time"][time_mask])
    data.create_index_map("stack", stack_map)
    data.create_reverse_map("stack", stack_rmap)
    data.create_index_map("xyz", np.array(["x", "y", "z"]))
    data.create_index_map("grid_xy", np.array(["x", "y"]))
    data.create_index_map("dish", np.arange(len(info["dish_positions"])))
    data.create_index_map(
        "threshold",
        np.arange(info["time_datasets"]["rfi_frame_excision_threshold"].shape[1]),
    )
    data.create_index_map("config", np.arange(len(info["config_json"])))
    data.create_index_map("file", np.array([os.path.basename(f) for f in info["files"]]))
    if "eval" in arrays or "evec" in arrays:
        data.create_index_map("ev", np.arange(info["n_ev"]))
    if "radiometer_chi2" in arrays:
        data.create_index_map("pol_product", np.array(["XX", "XY", "YY"]))

    def _add(name, arr, axes):
        dset = data.create_dataset(name, data=arr)
        dset.attrs["axis"] = np.array(axes)

    for name, arr in arrays.items():
        if name == "flags/element_flags" and name not in dsets:
            continue
        _add(name, arr, CORR_DATASETS[name][1])

    if "flags/inputs" in dsets:
        eflags = arrays["flags/element_flags"]
        inputs = (eflags > 0).any(axis=0) if nfreq else np.zeros((ninput, ntime), bool)
        _add("flags/inputs", inputs.astype(np.float32), ("input", "time"))

    if "digital_gain" in dsets:
        _add("digital_gain", gain.astype(np.complex64), ("freq", "input"))

    for name in TIME_DATASETS:
        arr = info["time_datasets"][name][time_mask]
        _add(name, arr, ("time", "threshold") if arr.ndim == 2 else ("time",))
    _add("file_index", info["file_index"][time_mask], ("time",))

    input_axes = {
        "feed_pos_disp_m": ("input", "xyz"),
        "feed_positions_m": ("input", "xyz"),
        "main_array_grid_indices": ("input", "grid_xy"),
    }
    for name, arr in info["input_tables"].items():
        _add(f"input_info/{name}", arr[input_ind], input_axes.get(name, ("input",)))

    _add("dish_positions_in_grid_coords", info["dish_positions"], ("dish", "xyz"))
    _add("config_json", np.array(info["config_json"], dtype=bytes), ("config",))

    # Attributes: those of the first file, updated for the selection
    for key, val in info["attrs"].items():
        data.attrs[key] = val
    data.attrs["acquisition_type"] = "corr"
    data.attrs["andata_version"] = ANDATA_VERSION
    data.attrs["acquisition"] = info["acquisition"]
    data.attrs["num_elements"] = ninput
    data.attrs["num_prod"] = nprod
    data.attrs["num_file_f"] = nfreq
    data.attrs["digital_gains_applied"] = bool(apply_gain and info["has_gains"])
    data.attrs["digital_gain_norm"] = info["gain_norm"]
    data.attrs["crs_board_remap"] = np.array(info["crs_board_remap"])
    data.attrs["crs_board_remap_source"] = info["crs_board_remap_source"]

    return data


# Private Functions
# -----------------

# Utilities


def _scan_xengine_files(files, normalise_gain=True, crs_board_remap=None):
    """Read the metadata of a set of X-engine files and check they are compatible.

    Parameters
    ----------
    files : list of str
        The files.
    normalise_gain : bool
        Compute the digital gain normalisation.
    crs_board_remap : list of int, optional
        Override the CRS board remap recorded in the files.

    Returns
    -------
    info : dict
        The sorted file names, the filled time samples of each file, the per-time
        datasets, the layout of the files, their static datasets and attributes,
        and what is needed to apply the digital gains.

    Raises
    ------
    AnDataError
        If the files are not X-engine files, have different layouts, or their
        times are not increasing.
    """
    files = [os.path.abspath(f) for f in files]
    if not files:
        raise ValueError("File list is empty.")

    per_file = []
    for fname in files:
        with h5py.File(fname, "r") as fh:
            if typeutils.bytes_to_unicode(fh.attrs.get("file_mode", "")) != "CHORD":
                raise AnDataError(f"{fname} is not a CHORD X-engine file.")
            per_file.append(
                {
                    "abs_file_idx": int(fh.attrs["abs_file_idx"]),
                    "time_datasets": {name: fh[name][:] for name in TIME_DATASETS},
                    "prod": fh["index_map/prod"][:],
                    "label": [typeutils.bytes_to_unicode(x) for x in fh["index_map/label"][:]],
                    "freq": fh["index_map/freq"][:],
                    "gain_key": _digital_gain_key(fh),
                }
            )

    order = np.argsort([i["abs_file_idx"] for i in per_file], kind="stable")
    files = [files[i] for i in order]
    per_file = [per_file[i] for i in order]

    first = per_file[0]
    for fname, finfo in zip(files, per_file):
        if (
            not np.array_equal(finfo["prod"], first["prod"])
            or finfo["label"] != first["label"]
            or not np.array_equal(finfo["freq"], first["freq"])
        ):
            raise AnDataError(
                f"File {fname} has a different frequency, product or input "
                f"layout from {files[0]}."
            )

    # Keep only the filled time bins
    valid = [i["time_datasets"]["time_center_t_inst_ns"] != 0 for i in per_file]
    time_datasets = {
        name: np.concatenate([i["time_datasets"][name][v] for i, v in zip(per_file, valid)])
        for name in TIME_DATASETS
    }
    t_ns = time_datasets["time_center_t_inst_ns"]
    if len(t_ns) == 0:
        raise AnDataError("No filled time samples in any file.")
    if np.any(np.diff(t_ns) <= 0):
        raise AnDataError("Times are not strictly increasing across the files.")

    with h5py.File(files[0], "r") as fh:
        attrs = {k: fh.attrs[k] for k in fh.attrs if k not in INPUT_ATTRS + ("input_list",)}
        input_list = fh.attrs["input_list"][:]
        input_tables = {name: fh["index_map"][name][:] for name in INPUT_TABLES}
        input_tables["label"] = np.array(
            [typeutils.bytes_to_unicode(x) for x in input_tables["label"]]
        )
        for name in INPUT_ATTRS:
            input_tables[name] = fh.attrs[name][:]
        dish_positions = fh["index_map/dish_positions_in_grid_coords"][:]
        config_json = [bytes(x) for x in fh["config_json"][:]]
        available = [name for name, (raw, _) in CORR_DATASETS.items() if raw in fh]
        dtypes = {name: fh[raw].dtype for name, (raw, _) in CORR_DATASETS.items() if raw in fh}
        n_ev = fh["eval"].shape[1] if "eval" in fh else 0
        has_gains = "digital_gains" in fh
        file_remap = get_crs_board_remap(fh)

    # Which CRS board feeds each slot: given, recorded in the files, or default
    if crs_board_remap is not None:
        remap, remap_source = list(crs_board_remap), "argument"
    elif file_remap is not None:
        remap, remap_source = file_remap, "config_json"
        with h5py.File(files[-1], "r") as fh:
            last_remap = get_crs_board_remap(fh)
        if last_remap is not None and last_remap != file_remap:
            raise AnDataError("The crs_board_remap changes within the files.")
    else:
        remap, remap_source = list(CRS_BOARD_REMAP), "default"
        logger.info(
            "The files do not record the crs_board_remap, using the default "
            f"{list(CRS_BOARD_REMAP)}."
        )

    fengine_input = digital_gain_input(input_list, remap)

    gain_norm = 1.0
    if has_gains and normalise_gain:
        with h5py.File(files[0], "r") as fh:
            g = np.abs(read_digital_gains(fh, first["freq"]["centre"], fengine_input))
        g = g[:, input_tables["type"] == INPUT_TYPE_DISH]
        gain_norm = float(np.median(g[g > 0])) if np.any(g > 0) else 1.0

    gain_keys = [i["gain_key"] for i in per_file]
    if len(set(gain_keys)) > 1:
        logger.warning(
            f"The digital gains change within the files ({len(set(gain_keys))} "
            "different sets). Each file is corrected with its own gains, but only "
            "the first set is saved in `digital_gain`."
        )

    if has_gains:
        available += ["digital_gain"]
    available += ["flags/inputs"]

    return {
        "files": files,
        "acquisition": os.path.basename(os.path.dirname(files[0])),
        "freq": first["freq"],
        "prod": first["prod"],
        "valid": valid,
        "time": t_ns * 1e-9,
        "time_datasets": time_datasets,
        "file_index": np.concatenate(
            [np.full(v.sum(), ii, dtype=np.int32) for ii, v in enumerate(valid)]
        ),
        "attrs": attrs,
        "input_list": input_list,
        "input_tables": input_tables,
        "dish_positions": dish_positions,
        "config_json": config_json,
        "available": available,
        "dtypes": dtypes,
        "n_ev": n_ev,
        "has_gains": has_gains,
        "crs_board_remap": remap,
        "crs_board_remap_source": remap_source,
        "fengine_input": fengine_input,
        "gain_norm": gain_norm,
        "gain_keys": gain_keys,
    }


def _digital_gain_key(fh):
    """A key identifying the gains in a file (to detect when they change)."""
    if "digital_gains" not in fh:
        return None
    dg = fh["digital_gains"]
    upd = int(dg.attrs.get("selected_update_idx", 0))
    return (
        typeutils.bytes_to_unicode(dg.attrs.get("acquisition_name", "")),
        typeutils.bytes_to_unicode(dg["update_id"][upd]),
        float(dg["index_map/update_time"][upd]),
    )


def _input_map(info, input_ind):
    """The `input` index map, (`chan_id`, `correlator_input`), for some inputs."""
    imap = np.empty(
        len(input_ind), dtype=[("chan_id", "<u2"), ("correlator_input", "U32")]
    )
    imap["chan_id"] = info["input_list"][input_ind]
    imap["correlator_input"] = info["input_tables"]["label"][input_ind]
    return imap


def _resolve_time_sel(info, start, stop):
    """For each file, a boolean mask of the time bins to read.

    `start` and `stop` index the filled time samples of all the files together,
    with the same conventions as a python slice.
    """
    ntime = len(info["time"])
    start, stop, _ = slice(start, stop).indices(ntime)
    keep = np.zeros(ntime, dtype=bool)
    keep[start:stop] = True

    time_sel = []
    t0 = 0
    for valid in info["valid"]:
        mask = np.zeros(len(valid), dtype=bool)
        nt = int(valid.sum())
        mask[np.flatnonzero(valid)] = keep[t0 : t0 + nt]
        time_sel.append(mask)
        t0 += nt
    return time_sel


def _resolve_datasets(datasets, available):
    """Get the list of analysis format datasets to read.

    Parameters
    ----------
    datasets : list of str or None
        Requested datasets (analysis or raw names). None for the default.
    available : list of str
        Datasets that can be read from the files.

    Returns
    -------
    datasets : list of str
        Analysis format names of the datasets to read.
    """
    if datasets is None:
        return [d for d in available if d not in PLACEHOLDER_DATASETS]

    out = []
    for name in datasets:
        name = name if name in CORR_DATASETS else _RAW_ALIASES.get(name, name)
        if name in available:
            out.append(name)
        else:
            logger.debug(f"Dataset {name} not in the files, skipping.")
    return out


def _walk_datasets(group, root=""):
    """Names of all the datasets in a container (including those in groups)."""
    names = []
    for key, item in group.items():
        name = f"{root}{key}"
        if memdata.is_group(item):
            if name not in ("index_map", "reverse_map", "history", "cal"):
                names += _walk_datasets(item, f"{name}/")
        else:
            names.append(name)
    return names


def _ensure_file_names(files):
    """Turn a filename, glob, open file, or list of them into a list of names."""
    if isinstance(files, str | h5py.File):
        files = [files]

    out = []
    for f in files:
        if isinstance(f, h5py.File):
            out.append(f.filename)
        else:
            out += sorted(glob.glob(f)) if glob.has_magic(f) else [f]
    if not out:
        raise ValueError("Acquisition file list is empty.")
    return out


def _open_files(files, opened):
    """Ensure that files are open, keeping a record of what was done.

    The arguments are modified in-place instead of returned, so that partial
    work is recorded in the event of an error.
    """
    for ii, this_file in enumerate(list(files)):
        # Sort out how to get an open hdf5 file.
        open_file, was_opened = memdata.get_file(this_file, mode="r")
        opened[ii] = was_opened
        files[ii] = open_file


def _get_dataset_names(f):
    """Get the analysis format names of the datasets in an X-engine file.

    Parameters
    ----------
    f : filename or h5py.File
        An X-engine file.

    Returns
    -------
    dataset_names : tuple of str
        Names of the datasets that can be read (see :data:`CORR_DATASETS`).
    """
    f, toclose = memdata.get_file(f, mode="r")
    try:
        dataset_names = tuple(
            name for name, (raw, _) in CORR_DATASETS.items() if raw in f
        )
        if "digital_gains" in f:
            dataset_names += ("digital_gain",)
        dataset_names += ("flags/inputs",)
    finally:
        if toclose:
            f.close()
    return dataset_names


def _ensure_1D_selection(selection):
    if isinstance(selection, tuple):
        if len(selection) != 1:
            raise ValueError("Wrong number of indices.")
        selection = selection[0]
    if selection is None:
        selection = np.s_[:]
    elif hasattr(selection, "__iter__"):
        selection = np.array(selection)
    elif isinstance(selection, slice):
        pass
    elif np.issubdtype(type(selection), np.integer):
        selection = np.s_[selection : selection + 1]
    else:
        raise ValueError("Cannont be converted to a 1D selection.")

    if isinstance(selection, np.ndarray):
        if selection.ndim != 1:
            raise ValueError("Data selections may only be one dimensional.")
        # The following is more efficient and solves h5py issue #425. Converts
        # to integer selection.
        if len(selection) == 1:
            return _ensure_1D_selection(selection[0])
        if np.issubdtype(selection.dtype, np.integer):
            if np.any(np.diff(selection) <= 0):
                raise ValueError("h5py requires sorted non-duplicate selections.")
        elif not np.issubdtype(selection.dtype, bool):
            raise ValueError("Array selections must be integer or boolean type.")
        elif np.issubdtype(selection.dtype, bool):
            # This is a workaround for h5py/h5py#1750
            selection = selection.nonzero()[0]

    return selection


def _convert_to_slice(selection):
    if hasattr(selection, "__iter__") and len(selection) > 1:
        uniq_step = np.unique(np.diff(selection))

        if (len(uniq_step) == 1) and uniq_step[0]:
            a = selection[0]
            b = selection[-1]
            b = b + (1 - (b < a) * 2)

            selection = slice(a, b, uniq_step[0])

    return selection


def _resolve_stack_prod_input_sel(
    stack_sel, stack_map, stack_rmap, prod_sel, prod_map, input_sel, input_map
):
    nsels = (stack_sel is not None) + (prod_sel is not None) + (input_sel is not None)
    if nsels > 1:
        raise ValueError(
            "Only one of *stack_sel*, *input_sel*, and *prod_sel* may be specified."
        )

    if nsels == 0:
        stack_sel = _ensure_1D_selection(stack_sel)
        prod_sel = _ensure_1D_selection(prod_sel)
        input_sel = _ensure_1D_selection(input_sel)
    else:
        if prod_sel is not None:
            prod_sel = _ensure_1D_selection(prod_sel)
            # Choose inputs involved in selected products.
            input_sel = _input_sel_from_prod_sel(prod_sel, prod_map)
            stack_sel = _stack_sel_from_prod_sel(prod_sel, stack_rmap)
        elif input_sel is not None:
            input_sel = _ensure_1D_selection(input_sel)
            prod_sel = _prod_sel_from_input_sel(input_sel, input_map, prod_map)
            stack_sel = _stack_sel_from_prod_sel(prod_sel, stack_rmap)
        else:  # stack_sel
            stack_sel = _ensure_1D_selection(stack_sel)
            prod_sel = _prod_sel_from_stack_sel(stack_sel, stack_map, stack_rmap)
            input_sel = _input_sel_from_prod_sel(prod_sel, prod_map)

        # Now we need to rejig the index maps for the subsets of the inputs,
        # prods.
        stack_inds = np.arange(len(stack_map), dtype=int)[stack_sel]
        input_inds = np.arange(len(input_map), dtype=int)[input_sel]

        stack_rmap = stack_rmap[prod_sel]
        stack_rmap["stack"] = _search_array(stack_inds, stack_rmap["stack"])

        # Remake stack map from scratch, since prod referenced in current stack
        # map may have dissapeared.
        stack_map = np.empty(len(stack_inds), dtype=stack_map.dtype)
        stack_map["prod"] = _search_array(
            stack_rmap["stack"], np.arange(len(stack_inds))
        )
        stack_map["conjugate"] = stack_rmap["conjugate"][stack_map["prod"]]

        prod_map = prod_map[prod_sel]
        pa = _search_array(input_inds, prod_map["input_a"])
        pb = _search_array(input_inds, prod_map["input_b"])
        prod_map["input_a"] = pa
        prod_map["input_b"] = pb
        input_map = input_map[input_sel]
    return stack_sel, stack_map, stack_rmap, prod_sel, prod_map, input_sel, input_map


def _search_array(a, v):
    """Find the indices in array `a` of values in array 'v'.

    Use algorithm that presorts `a`, efficient if `v` is long.
    """
    a_sort_inds = np.argsort(a, kind="mergesort")
    a_sorted = a[a_sort_inds]
    indeces_in_sorted = np.searchsorted(a_sorted, v)
    # Make sure values actually present.
    if not np.all(v == a_sorted[indeces_in_sorted]):
        raise ValueError("Element in 'v' not in 'a'.")
    return a_sort_inds[indeces_in_sorted]


def _input_sel_from_prod_sel(prod_sel, prod_map):
    prod_map = prod_map[prod_sel]
    input_sel = []
    for p0, p1 in prod_map:
        input_sel.append(p0)
        input_sel.append(p1)
    # ensure_1D here deals with h5py issue #425.
    return _ensure_1D_selection(sorted(set(input_sel)))


def _prod_sel_from_input_sel(input_sel, input_map, prod_map):
    inputs = list(np.arange(len(input_map), dtype=int)[input_sel])
    prod_sel = []
    for ii, p in enumerate(prod_map):
        if p[0] in inputs and p[1] in inputs:
            prod_sel.append(ii)
    # ensure_1D here deals with h5py issue #425.
    return _ensure_1D_selection(prod_sel)


def _stack_sel_from_prod_sel(prod_sel, stack_rmap):
    stack_sel = stack_rmap["stack"][prod_sel]
    return _ensure_1D_selection(sorted(set(stack_sel)))


def _prod_sel_from_stack_sel(stack_sel, stack_map, stack_rmap):
    stack_inds = np.arange(len(stack_map))[stack_sel]
    stack_rmap_sort_inds = np.argsort(stack_rmap["stack"], kind="mergesort")
    stack_rmap_sorted = stack_rmap["stack"][stack_rmap_sort_inds]
    left_indeces = np.searchsorted(stack_rmap_sorted, stack_inds, side="left")
    right_indeces = np.searchsorted(stack_rmap_sorted, stack_inds, side="right")
    prod_sel = []
    for ii in range(len(stack_inds)):
        prod_sel.append(stack_rmap_sort_inds[left_indeces[ii] : right_indeces[ii]])
    prod_sel = np.concatenate(prod_sel)
    return _ensure_1D_selection(sorted(set(prod_sel)))
