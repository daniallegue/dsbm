"""Precompute the min-flux metastable state pair (pi_0, pi_1) for one tetrapeptide's
TPS task, replicating mdgen's tps_inference.py::do()'s state-selection logic exactly
(same MSM fit, same flux-argmin state pair, same random seed) but as a standalone,
cacheable step instead of being embedded in their inference loop.

Must be run with a Python that has `pyemma` importable -- e.g. the `tps_eval` conda
env (see vendor/mdgen/PROVENANCE.md), NOT the project's main venv.

Usage (run from the repo root):
    <tps_eval python> scripts/precompute_tps_states.py --peptide IPGD \
        --mddir data/4AA_sims --cache_dir data/tetrapeptide_tps/IPGD

Writes two files to --cache_dir:
  {peptide}_metadata.pkl  -- msm/cmsm/tica/pcca/kmeans/ref_kmeans (pyemma objects),
                              in the exact schema tps_inference.do() writes, so
                              vendor/mdgen/scripts/analyze_peptide_tps.py can consume
                              it unmodified once copied into a sample output dir.
  {peptide}_states.npz     -- start_idxs, end_idxs, start_state, end_state as plain
                              numpy, so the (pyemma-free) training-side dataset loader
                              can read it without needing pyemma installed.
"""
import argparse
import contextlib
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "vendor", "mdgen"))

import mdgen.analysis  # noqa: E402


@contextlib.contextmanager
def temp_seed(seed):
    # verbatim from mdgen/tps_inference.py -- seeds numpy's global RNG for the
    # duration of the MSM fit, matching upstream's reproducibility convention.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def compute_states(peptide, mddir, seed=137, nstates=10):
    with temp_seed(seed):
        feats, ref = mdgen.analysis.get_featurized_traj(f"{mddir}/{peptide}/{peptide}", sidechains=True)
        tica, _ = mdgen.analysis.get_tica(ref)
        kmeans, ref_kmeans = mdgen.analysis.get_kmeans(tica.transform(ref))
        msm, pcca, cmsm = mdgen.analysis.get_msm(ref_kmeans, nstates=nstates)

    # min-flux metastable pair -- verbatim logic from tps_inference.py::do()
    flux_mat = cmsm.transition_matrix * cmsm.pi[None, :]
    flux_mat[flux_mat < 0.0000001] = np.inf
    start_state, end_state = np.unravel_index(np.argmin(flux_mat, axis=None), flux_mat.shape)
    ref_discrete = msm.metastable_assignments[ref_kmeans]
    start_idxs = np.where(ref_discrete == start_state)[0]
    end_idxs = np.where(ref_discrete == end_state)[0]
    if len(start_idxs) == 0 or len(end_idxs) == 0:
        raise RuntimeError(f"no start/end frames found for '{peptide}' (start_state={start_state}, "
                            f"end_state={end_state}) -- min-flux pair degenerate for this peptide")

    metadata = {"msm": msm, "cmsm": cmsm, "tica": tica, "pcca": pcca, "kmeans": kmeans, "ref_kmeans": ref_kmeans}
    states = {"start_idxs": start_idxs, "end_idxs": end_idxs,
              "start_state": int(start_state), "end_state": int(end_state)}
    return metadata, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peptide", required=True, help="e.g. IPGD")
    parser.add_argument("--mddir", required=True, help="dir containing {peptide}/{peptide}.pdb + .xtc")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--nstates", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    metadata_path = os.path.join(args.cache_dir, f"{args.peptide}_metadata.pkl")
    states_path = os.path.join(args.cache_dir, f"{args.peptide}_states.npz")

    metadata, states = compute_states(args.peptide, args.mddir, seed=args.seed, nstates=args.nstates)

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    np.savez(states_path, **states)

    print(f"start_state={states['start_state']} ({len(states['start_idxs'])} frames), "
          f"end_state={states['end_state']} ({len(states['end_idxs'])} frames)")
    print(f"wrote {metadata_path}")
    print(f"wrote {states_path}")


if __name__ == "__main__":
    main()
