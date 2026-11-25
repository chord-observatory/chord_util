# chord_util

General CHORD utilities.

- `chord_util.andata`: read the CHORD X-engine N² visibility files (kotekan
  `hdf5N2Write`) and concatenate them into an analysis format `CorrData`
  container, laid out like `ch_util.andata.CorrData`. Includes `CorrReader`
  for selecting data before reading, and the F-engine digital gain handling.
- `chord_util.rfi`: RFI tools (placeholder for the static RFI mask).

## Installation

```sh
pip install -e .
```

It needs `caput`, `h5py`, `bitshuffle`, `numpy` and `mpi4py` (for distributed
reads). It does not depend on any CHIME packages.

## Example

```python
from chord_util import andata

reader = andata.CorrReader("/path/to/acq_20260911_232919_986789057/vis_*.h5")
reader.select_input_type("dish")       # drop the RFI monitors
reader.select_freq_range(600.0, 700.0)  # MHz
data = reader.read()                    # andata.CorrData
```

The pipeline tasks that use it (`QueryAcquisitionFiles`, `LoadCorrDataFiles`)
are in `chord_pipeline.core.io`.

## Tests

The tests use small synthetic X-engine files made by `tests/xengine_testdata.py`
(no real data needed):

```sh
pip install -e ".[test]"
python -m pytest            # serial
mpirun -np 3 python -m pytest   # distributed reads under MPI
```
