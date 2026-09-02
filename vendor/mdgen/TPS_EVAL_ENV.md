# `tps_eval` conda environment

`mdgen/analysis.py` (vendored unmodified, see PROVENANCE.md) hard-imports `pyemma`,
which has no Windows wheels and fails to build from source against MSVC. Its
TICA/k-means/MSM fitting (used by `scripts/precompute_tps_states.py`) and the final
`scripts/analyze_peptide_tps.py` eval script both need to run under a separate conda
env with a prebuilt `pyemma` from conda-forge, instead of the project's main
pip venv.

## Setup

```
"%USERPROFILE%\miniforge3\Scripts\conda.exe" create -n tps_eval -c conda-forge python=3.11 pyemma mdtraj pandas -y
```

Requires Miniforge installed at `%USERPROFILE%\miniforge3` (user-space, no
admin/BIOS needed -- installed via the Miniforge3-Windows-x86_64.exe installer with
`/InstallationType=JustMe /RegisterPython=0 /AddToPath=0`).

### Two required fixes after creating the env

**1. Always run with `PATH` including the env's `Library\bin`.** Invoking
`envs\tps_eval\python.exe` directly (without activating) leaves conda-forge's DLLs
(zlib, xdrfile, etc., used by mdtraj's compiled readers) off `PATH`, which crashes
native calls like `mdtraj.load(...)` with exit code `0xC06D007F` and zero Python
traceback. Always prepend, in this order, before invoking python:
`envs\tps_eval`, `envs\tps_eval\Library\mingw-w64\bin`, `envs\tps_eval\Library\usr\bin`,
`envs\tps_eval\Library\bin`, `envs\tps_eval\Scripts`, `miniforge3\condabin`.
(Equivalently: use `conda activate tps_eval` or `conda run -n tps_eval` if invoking
from a shell conda itself was set up in.)

**2. Always run with `-s`** (disable user site-packages). A stray
`pip install --user numpy` elsewhere on this machine
(`%APPDATA%\Roaming\Python\Python311\site-packages`) otherwise shadows the conda
env's own numpy with a build that's ABI-incompatible with conda-forge's compiled
`pyemma`/`mdtraj` extensions, causing the same kind of silent native crash as (1).

**3. Two one-line patches to pyemma itself** (installed package, not vendored code --
pyemma 2.5.12 was last released in 2022 and has known incompatibilities with modern
numpy/mdtraj that show up the moment you actually run a featurizer, not just import
it):

- `pyemma/coordinates/data/featurization/angles.py`, `SideChainTorsions.__init__`:
  `np.vstack(valid.values())` -> `np.vstack(list(valid.values()))` (modern numpy's
  `vstack` requires an actual sequence, not a `dict_values` view; behavior is
  identical either way).
- `pyemma/coordinates/data/feature_reader.py`, `FeatureReader.__init__`:
  `mdtraj.version.version` -> `mdtraj.__version__` (modern mdtraj no longer exposes
  the `mdtraj.version` submodule).
- Bare deprecated numpy scalar aliases removed in numpy >=1.24 (`np.bool`/`np.float`,
  as opposed to the still-valid `np.bool_`/`np.float64`/builtin `bool`/`float`):
  `pyemma/_ext/variational/estimators/moments.py` (2 call sites, `np.bool` ->
  `np.bool_`), `.../tests/test_moments.py` (3 sites), `.../covar_c/covartools.py`
  (3 sites, `numpy.bool` -> `numpy.bool_`), `pyemma/msm/estimators/lagged_model_validators.py`
  (`np.float` -> `float`), `pyemma/msm/estimators/_OOM_MSM.py` (`np.float` ->
  `float`). Ran `grep -rnE "numpy\.(bool|int|float|object|complex|str)\b|np\.(bool|int|float|object|complex|str)\b" --include=*.py` over the whole `pyemma` package to
  confirm this was the complete set (one remaining hit is an inert comment).

None of these patches change any numerical behavior -- all are purely fixing
pyemma's use of APIs that no longer exist / no longer accept the same input shape in
the numpy/mdtraj versions available today. If setting up this env fresh, re-apply
all of them before running anything that calls TICA/MSM fitting or
`get_featurized_traj(..., sidechains=True)`.

## Verifying the env

```
"%USERPROFILE%\miniforge3\envs\tps_eval\python.exe" -s -c "import pyemma, mdtraj; print(pyemma.__version__, mdtraj.__version__)"
```
(with the `Library\bin` PATH prefix from fix (1) above, or the featurizer-level calls
will crash rather than raise a clean ImportError).
