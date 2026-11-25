"""Layout and feed mask of the CHORD pathfinder.

CHIME keeps its layout in a database (``ch_util.layout``). The pathfinder does not
have one yet, so these are two small text files kept with the package:

- ``data/pathfinder_layout.txt``: the inputs (label, channel id, east and north
  positions in metres, polarisation), one row per input, in the order of the
  X-engine data. This does not change from night to night. It is made from the
  ``feed_positions_m`` attribute of an X-engine file (:func:`layout_from_file`,
  :func:`write_layout`), which has the positions of the X-engine configuration at
  full precision. :func:`check_layout` compares it with another file.
- ``data/pathfinder_feedmask.yaml``: the history of the bad inputs. Each entry holds
  from its ``start`` time (UTC) until the next one (:func:`bad_inputs_at`). A new
  entry is added after the inspection of a night (:func:`update_feedmask`).

The pointing is a setting of the acquisition, not of the layout. It is in the
attributes of the data files (``dish_coelev_deg``, see :func:`pointing_dec`).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

_DATA = Path(__file__).parent / "data"
LAYOUT_FILE = _DATA / "pathfinder_layout.txt"
FEEDMASK_FILE = _DATA / "pathfinder_feedmask.yaml"

_LAYOUT_DTYPE = [
    ("label", "U8"),
    ("chan_id", "u2"),
    ("ew_m", "f8"),
    ("ns_m", "f8"),
    ("pol", "U1"),
]


def _labels(fh):
    return np.array(
        [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in fh["index_map/label"][:]
        ]
    )


def layout_from_file(fname):
    """The layout recorded in an X-engine file.

    Parameters
    ----------
    fname : str
        An X-engine N2 file (``vis_*.h5``).

    Returns
    -------
    layout : np.ndarray
        One entry per input, in the order of the data: ``label``, ``chan_id``,
        ``ew_m``, ``ns_m``, ``pol``.
    """
    with h5py.File(fname, "r") as fh:
        labels = _labels(fh)
        pos = np.asarray(fh.attrs["feed_positions_m"], dtype=np.float64)
        pol = np.asarray(fh["index_map/pol"][:]).astype(int)

    out = np.zeros(len(labels), dtype=_LAYOUT_DTYPE)
    out["label"] = labels
    out["chan_id"] = np.arange(len(labels))
    out["ew_m"] = pos[:, 0]
    out["ns_m"] = pos[:, 1]
    out["pol"] = np.array(["X", "Y"])[pol]
    return out


def write_layout(fname, outfile=LAYOUT_FILE):
    """Write the layout of an X-engine file as a text file.

    The positions are written at full precision.
    """
    layout = layout_from_file(fname)
    with open(outfile, "w") as fh:
        fh.write(
            "# CHORD pathfinder layout: one row per input, "
            "in the order of the X-engine data.\n"
        )
        src = f"{Path(fname).parent.name}/{Path(fname).name}"
        fh.write(f"# From feed_positions_m of {src} (east, north; metres).\n")
        fh.write("# label chan_id ew_m ns_m pol\n")
        for row in layout:
            ew, ns = float(row["ew_m"]), float(row["ns_m"])
            fh.write(f"{row['label']} {row['chan_id']} {ew!r} {ns!r} {row['pol']}\n")


def read_layout(path=None):
    """Read a layout text file (default: the pathfinder layout of this package)."""
    return np.genfromtxt(
        path or LAYOUT_FILE, dtype=_LAYOUT_DTYPE, encoding=None, comments="#", ndmin=1
    )


def check_layout(fname, path=None, tol=1e-6):
    """Differences between the layout and that of an X-engine file.

    Returns
    -------
    problems : list of str
        Empty if the labels, order, polarisations and positions (within ``tol``
        metres) are the same.
    """
    a, b = read_layout(path), layout_from_file(fname)
    if len(a) != len(b):
        return [f"{len(a)} inputs in the layout, {len(b)} in the file"]
    out = [
        f"input {i}: {a['label'][i]} / {b['label'][i]}"
        for i in np.flatnonzero(a["label"] != b["label"])
    ]
    out += [f"pol of {a['label'][i]}" for i in np.flatnonzero(a["pol"] != b["pol"])]
    off = np.hypot(a["ew_m"] - b["ew_m"], a["ns_m"] - b["ns_m"])
    out += [
        f"position of {a['label'][i]} differs by {off[i]:.2e} m"
        for i in np.flatnonzero(off > tol)
    ]
    return out


def pointing_dec(attrs):
    """Declination [deg] of the pointing.

    From the attributes of an X-engine file or timestream.
    """
    return float(attrs["itrs_lat_deg"] + attrs["dish_coelev_deg"])


def _as_utc(t):
    if isinstance(t, datetime.datetime):
        t = t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)
        return t.astimezone(datetime.timezone.utc)
    if isinstance(t, str):
        return _as_utc(datetime.datetime.fromisoformat(t.replace("Z", "+00:00")))
    return datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc)


def read_feedmask(path=None):
    """The entries of a feed mask file, sorted by start time."""
    with open(path or FEEDMASK_FILE) as fh:
        entries = yaml.safe_load(fh) or []
    return sorted(entries, key=lambda e: _as_utc(e["start"]))


def bad_inputs_at(time, path=None):
    """The bad inputs at a time.

    Parameters
    ----------
    time : float, str or datetime
        UNIX time, ISO string, or datetime (UTC if naive).
    path : str, optional
        Feed mask file. Default the one of this package.

    Returns
    -------
    bad : dict
        Label of each bad input and the reason for it.
    """
    t = _as_utc(time)
    valid = [e for e in read_feedmask(path) if _as_utc(e["start"]) <= t]
    if not valid:
        raise ValueError(f"No feed mask entry at or before {t.isoformat()}")
    return dict(valid[-1]["bad_inputs"])


def update_feedmask(start, bad_inputs, acquisition=None, comment=None, path=None):
    """Add an entry to the feed mask file.

    An entry with the same start time is replaced.

    Parameters
    ----------
    start : float, str or datetime
        Time from which the entry holds.
    bad_inputs : dict
        Label -> reason.
    acquisition, comment : str, optional
        Where the entry comes from.
    path : str, optional
        Feed mask file. Default the one of this package.
    """
    path = path or FEEDMASK_FILE
    start = _as_utc(start)
    entries = read_feedmask(path) if Path(path).exists() else []
    entries = [e for e in entries if _as_utc(e["start"]) != start]
    entry = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "acquisition": acquisition,
        "comment": comment,
    }
    entry["bad_inputs"] = {str(k): str(v) for k, v in sorted(bad_inputs.items())}
    entries.append(entry)
    entries.sort(key=lambda e: _as_utc(e["start"]))
    with open(path, "w") as fh:
        fh.write(
            "# CHORD pathfinder bad inputs: each entry holds from its start time "
            "(UTC) until the next one.\n"
        )
        fh.write("# Written by chord_util.layout.update_feedmask.\n")
        yaml.safe_dump(entries, fh, sort_keys=False)
