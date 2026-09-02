# Vendored from bjing2016/mdgen

Source: https://github.com/bjing2016/mdgen
Commit: 81482a403b91c5a8437046da1d8d321ba97089cc
License: MIT (see LICENSE in this directory)

Paper: Jing, Stärk, Jaakkola, Berger. "Generative Modeling of Molecular Dynamics
Trajectories." arXiv:2409.17808.

## What's vendored, and why

Only the data-prep and evaluation infrastructure needed for the Transition Path
Sampling (TPS) task, kept byte-for-byte identical to upstream (with one dead-code
fix, see below) so results stay comparable to the paper. Only a subset of the
original repo's files is copied in, and empty `__init__.py` files were added to
`mdgen/` and `scripts/` (upstream relies on implicit namespace packages) purely so
the two subdirectories resolve as regular packages.

**Deviations from upstream** (both are compatibility fixes for running 2021-2022-era
code against this project's modern numpy/torch, not logic changes):

- `scripts/prep_sims.py`'s non-`--atlas` `do_job` (both branches, actually)
  reference `args.atlas_dir`, which the script never defines as an argparse option
  (only `--sim_dir` exists, and is what the repo's own README example commands
  pass) -- as shipped, the script cannot run at all. Fixed both references to
  `args.sim_dir` to match; a dead-reference typo fix, not a change to any
  data-processing logic (stride, superposition, atom14 layout are all untouched).
- `mdgen/tensor_utils.py::batched_gather` (used by `geometry.py`'s
  atom14/atom37/torsion conversions) built a plain Python `list` of per-axis index
  arrays and indexed with it directly (`data[ranges]`), relying on numpy/torch's old
  convenience of treating such a list the same as a tuple of per-axis indices.
  Modern numpy raises `ValueError: setting an array element with a sequence...
  inhomogeneous shape` for this; modern torch still runs it but warns it will change
  meaning in PyTorch 2.9. Changed to `data[tuple(ranges)]` -- exactly the fix
  PyTorch's own deprecation warning names, and semantically identical to what "a
  list of per-axis index arrays" always meant. No change to what gets computed.

- `mdgen/geometry.py`, `rigid_utils.py`, `residue_constants.py`, `tensor_utils.py`
  -- raw MD (atom14/atom37) <-> (frames, torsions) conversions.
- `mdgen/protein.py`, `utils.py` -- `atom14_to_pdb` and friends, for writing
  generated trajectories back out to PDB.
- `mdgen/analysis.py` -- the eval harness: TICA / k-means / MSM fitting,
  discretization, transition-path likelihood scoring. Requires `pyemma`.
- `scripts/prep_sims.py` -- converts raw MD (`.xtc`/`.pdb`) into the `atom14`
  `.npy` files everything downstream reads. Requires `mdtraj`.
- `scripts/analyze_peptide_tps.py` -- the TPS scoring script, run unmodified
  against our generated trajectories at the end of the pipeline.
- `splits/4AA*.csv` -- the tetrapeptide name/seqres splits (100-peptide test set
  includes `IPGD`, the paper's Figure 3 TPS case study).

Explicitly not vendored: `mdgen/model/`, `mdgen/transport/`, `mdgen/wrapper.py`,
`mdgen/dataset.py`, `train.py`, `*_inference.py` -- all belong to mdgen's own
DiT+IPA generative model, which this project replaces with its own R-IMF
Riemannian bridge (see `bridge/sde/`, `bridge/data/tetrapeptide_tps.py`).
`mdgen.dataset.atom14_to_frames` is just a re-export of
`mdgen.geometry.atom14_to_frames`, so it's imported from `geometry` directly
instead of vendoring `dataset.py`.

## Running the vendored scripts

Run from this directory (`vendor/mdgen/`), matching upstream's own convention:

```
python -m scripts.prep_sims --split splits/4AA_test.csv --sim_dir <raw MD dir> --outdir <out dir> --suffix _i100 --stride 100
python -m scripts.analyze_peptide_tps --mddir <raw MD dir> --pdbdir <generated samples dir> --outdir <results dir> --plot --save --pdb_id IPGD
```

`scripts/analyze_peptide_tps.py` and the TICA/k-means/MSM parts of
`mdgen/analysis.py` need `pyemma`, which has no Windows wheels and fails to
build from source against MSVC. Run those specifically from the `tps_eval`
conda env (`pyemma`, `mdtraj` from conda-forge) instead of the project's main
venv -- see the repo README for setup.

## `scripts/analyze_peptide_tps.py`: replica baseline made optional

This is a real, deliberate deviation (not a compatibility shim) -- flagged here
explicitly since the whole point of vendoring was to keep the eval harness
untouched.

`--repdir` (an auxiliary "how does naive short replica MD compare" baseline,
computed via the same `mdgen.analysis` pipeline on a second independent MD
simulation of the same peptide) is not fetchable: it's not part of the public
`tetrapeptide-sims` HuggingFace dataset, and generating one means running
`scripts/run_peptide_sim.py` (not vendored here) -- a full explicit-solvent OpenMM
simulation that needs a GPU and `pymol`/`pdbfixer` to be practical. As shipped, the
script crashes immediately without `--repdir`, and it crashes *before* ever writing
the output `.pkl` -- so a missing replica baseline meant losing `gen_prob` /
`gen_valid_prob` / `gen_valid_rate` / `gen_JSD` too, the metrics that actually
compare against the reference MSM and don't conceptually depend on the replica at
all.

Changed `--repdir`'s default from `'share/4AA_sims_replica'` to `None`, and made the
entire replica section (MSM refit per replica-length, `rep_*`/`repcheat_*` output
keys, the two replica-dependent plot panels) conditional on `args.repdir` being set.
Behavior is byte-for-byte unchanged whenever `--repdir` *is* provided. The paper's
own aggregate replica-baseline numbers (Figure 3: 100ns MD JSD~0.24/valid~98%, 10ns
MD JSD~0.32/valid~90%, 1ns MD JSD~0.43/valid~70%, averaged over all 100 test
peptides) are cited directly as context instead of being reproduced locally.

## `scripts/analyze_peptide_tps.py`: generated-ensemble flux plot wrapped in try/except

Same failure shape as the replica issue above, found while smoke-testing against a
deliberately undertrained (20-iteration) checkpoint: `gen_tps_msm =
pyemma.msm.estimate_markov_model(list(gen_tp), lag=1)` and the `pyemma.msm.tpt(...)`
call right after it fit an MSM on the *generated* ensemble to draw a flux diagram
(`axs[1,3]`) -- purely diagnostic, not one of the `gen_prob`/`gen_valid_prob`/
`gen_valid_rate`/`gen_JSD` metrics, which are already computed earlier in `main()`
by this point. If the generated paths are few/short/low-quality enough that they
don't cleanly revisit both endpoint states within `gen_tps_msm`'s active set, this
raises a `KeyError` -- and, as shipped, that happens *before* the output `.pkl` is
ever written, so it silently loses the already-computed metrics too. Wrapped just
these two lines (`gen_tps_msm`/`gen_tpt` + the `plot_flux` call) in try/except;
on failure it prints a message and leaves that one subplot panel blank/titled
"skipped". No effect on any computed metric, and no effect on the reference-MSM
flux plot (`axs[0,3]`, always well-defined since start/end states were chosen from
the reference MSM in the first place).
