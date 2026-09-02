"""Batch-acquire more of MDGen's tetrapeptide MD corpus so bridge/data/tetrapeptide_tps_multi.py
has more than the 2 peptides (FLRH, IPGD) it amortizes over today.

Borrows MDGen's own data acquisition: the raw explicit-solvent simulations are published on
HuggingFace at https://huggingface.co/datasets/bjing-mit/tetrapeptide-sims (MIT license), laid
out as `4AA_sims/{peptide}/{peptide}.pdb` + `{peptide}.xtc` -- exactly the layout
vendor/mdgen/scripts/prep_sims.py already expects via --sim_dir, and exactly where the two
peptides already in this repo (data/4AA_sims/FLRH, IPGD) came from.

Per peptide, in order (see PROVENANCE.md / TPS_EVAL_ENV.md for why each step needs what it
needs):
  1. download {peptide}.pdb + {peptide}.xtc from HuggingFace -> --sim_dir
  2. convert to atom14 .npy via vendor/mdgen/scripts/prep_sims.py (this venv, needs mdtraj)
     -> --data_dir/{peptide}.npy
  3. precompute the min-flux TPS state pair via scripts/precompute_tps_states.py, run under the
     separate `tps_eval` conda env (needs pyemma -- see vendor/mdgen/TPS_EVAL_ENV.md; step 3
     reads the RAW .xtc directly via pyemma, so it must run BEFORE the raw file is deleted)
     -> --cache_dir/{peptide}/{peptide}_states.npz
  4. delete the raw .xtc (keep the tiny .pdb) to reclaim disk space, unless --keep_raw

One peptide is taken fully through all 4 steps before the next one starts, so disk usage never
exceeds one in-flight peptide's raw+converted footprint (~600MB) beyond whatever's already
completed -- deliberately, since this machine has limited free disk space. A peptide that fails
at any step is cleaned up (partial files removed) and skipped, with a warning printed, rather
than aborting the whole batch.

Usage (run from the repo root, in the project's main venv):
    python scripts/fetch_peptide_corpus.py --split vendor/mdgen/splits/4AA_train.csv --n_peptides 50

Requires the `tps_eval` conda env to already exist (see vendor/mdgen/TPS_EVAL_ENV.md); this
script locates it at %USERPROFILE%\\miniforge3 by default, override with --tps_eval_root.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_MDGEN = os.path.join(REPO_ROOT, "vendor", "mdgen")

HF_REPO_ID = "bjing-mit/tetrapeptide-sims"
HF_PREFIX = {"explicit": "4AA_sims", "implicit": "4AA_sims_implicit"}


def _tps_eval_python_and_env(tps_eval_root):
    env_root = os.path.join(tps_eval_root, "envs", "tps_eval")
    python = os.path.join(env_root, "python.exe")
    if not os.path.isfile(python):
        raise FileNotFoundError(
            f"tps_eval conda env not found at {env_root} -- see vendor/mdgen/TPS_EVAL_ENV.md "
            f"to create it, or pass --tps_eval_root if it's installed elsewhere")

    # Fixes (1)+(2) from TPS_EVAL_ENV.md: native mdtraj/pyemma calls crash with no traceback
    # unless the env's own DLL dirs are on PATH; -s (passed at call sites below) dodges a stray
    # `pip install --user numpy` that otherwise shadows the env's ABI-matched numpy.
    path_prefix = os.pathsep.join([
        env_root,
        os.path.join(env_root, "Library", "mingw-w64", "bin"),
        os.path.join(env_root, "Library", "usr", "bin"),
        os.path.join(env_root, "Library", "bin"),
        os.path.join(env_root, "Scripts"),
        os.path.join(tps_eval_root, "condabin"),
    ])
    env = os.environ.copy()
    env["PATH"] = path_prefix + os.pathsep + env.get("PATH", "")
    return python, env


def _already_done(name, data_dir, cache_dir):
    return (os.path.isfile(os.path.join(data_dir, f"{name}.npy"))
            and os.path.isfile(os.path.join(cache_dir, name, f"{name}_states.npz")))


def _free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def _download_sim(hf_api, name, prefix, sim_dir):
    from huggingface_hub import hf_hub_download
    peptide_dir = os.path.join(sim_dir, name)
    os.makedirs(peptide_dir, exist_ok=True)
    for ext in (".pdb", ".xtc"):
        hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset",
                         filename=f"{prefix}/{name}/{name}{ext}",
                         local_dir=sim_dir)
    # hf_hub_download mirrors the repo's own path layout under local_dir, i.e.
    # {sim_dir}/{prefix}/{name}/{name}{.pdb,.xtc} -- move up to {sim_dir}/{name}/... to match
    # what prep_sims.py / precompute_tps_states.py expect (no HF_PREFIX component).
    downloaded_dir = os.path.join(sim_dir, prefix, name)
    for ext in (".pdb", ".xtc"):
        src = os.path.join(downloaded_dir, f"{name}{ext}")
        dst = os.path.join(peptide_dir, f"{name}{ext}")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.move(src, dst)


def _convert_to_atom14(name, split_csv_row, sim_dir, data_dir):
    # prep_sims.py is invoked with cwd=VENDOR_MDGEN (matching its own documented convention,
    # PROVENANCE.md), so --sim_dir/--outdir/--split MUST be absolute -- a relative path here
    # would resolve against vendor/mdgen/ instead of the caller's cwd.
    sim_dir, data_dir = os.path.abspath(sim_dir), os.path.abspath(data_dir)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_split = os.path.abspath(os.path.join(tmp, "split.csv"))
        split_csv_row.to_csv(tmp_split)
        subprocess.run(
            [sys.executable, "-m", "scripts.prep_sims",
             "--split", tmp_split, "--sim_dir", sim_dir, "--outdir", data_dir],
            cwd=VENDOR_MDGEN, check=True)
    if not os.path.isfile(os.path.join(data_dir, f"{name}.npy")):
        raise RuntimeError(f"prep_sims.py ran but did not produce {data_dir}/{name}.npy")


def _precompute_states(name, sim_dir, cache_dir, tps_eval_root):
    python, env = _tps_eval_python_and_env(tps_eval_root)
    peptide_cache_dir = os.path.join(cache_dir, name)
    subprocess.run(
        [python, "-s", "scripts/precompute_tps_states.py",
         "--peptide", name, "--mddir", sim_dir, "--cache_dir", peptide_cache_dir],
        cwd=REPO_ROOT, env=env, check=True)


def fetch_one(name, split_df, sim_dir, data_dir, cache_dir, prefix, tps_eval_root, keep_raw):
    try:
        _download_sim(None, name, prefix, sim_dir)
        _convert_to_atom14(name, split_df.loc[[name]], sim_dir, data_dir)
        _precompute_states(name, sim_dir, cache_dir, tps_eval_root)
        if not keep_raw:
            xtc_path = os.path.join(sim_dir, name, f"{name}.xtc")
            if os.path.isfile(xtc_path):
                os.remove(xtc_path)
        return True, None
    except Exception as e:
        return False, e


def _cleanup_partial(name, sim_dir, data_dir, cache_dir, keep_raw):
    if not keep_raw:
        shutil.rmtree(os.path.join(sim_dir, name), ignore_errors=True)
    npy_path = os.path.join(data_dir, f"{name}.npy")
    if os.path.isfile(npy_path) and not os.path.isfile(
            os.path.join(cache_dir, name, f"{name}_states.npz")):
        # states step didn't finish -- leaving a converted .npy with no cache is harmless
        # (bridge/data/tetrapeptide_tps_multi.py requires both to consider a peptide available)
        # but delete it too so a re-run cleanly retries this peptide from scratch.
        os.remove(npy_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default=os.path.join(VENDOR_MDGEN, "splits", "4AA_train.csv"))
    parser.add_argument("--n_peptides", type=int, default=50, help="max NEW peptides to fetch this run")
    parser.add_argument("--sim_dir", default=os.path.join(REPO_ROOT, "data", "4AA_sims"))
    parser.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data", "4AA_data"))
    parser.add_argument("--cache_dir", default=os.path.join(REPO_ROOT, "data", "tetrapeptide_tps"))
    parser.add_argument("--variant", choices=list(HF_PREFIX), default="explicit")
    parser.add_argument("--tps_eval_root", default=os.path.expandvars(r"%USERPROFILE%\miniforge3"))
    parser.add_argument("--keep_raw", action="store_true", help="keep the raw .xtc after conversion")
    parser.add_argument("--min_free_gb", type=float, default=5.0,
                         help="stop starting new peptides once free disk space would drop below this")
    args = parser.parse_args()
    # Helper subprocesses run with a different cwd than this script's (prep_sims.py under
    # vendor/mdgen/, precompute_tps_states.py under the repo root) -- normalize to absolute
    # paths once here so a relative --sim_dir/--data_dir/--cache_dir resolves the same for both.
    args.sim_dir = os.path.abspath(args.sim_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    args.cache_dir = os.path.abspath(args.cache_dir)

    os.makedirs(args.sim_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    split_df = pd.read_csv(args.split, index_col="name")
    prefix = HF_PREFIX[args.variant]

    todo = [name for name in split_df.index if not _already_done(name, args.data_dir, args.cache_dir)]
    print(f"{len(split_df) - len(todo)}/{len(split_df)} peptides in '{args.split}' already done; "
          f"{len(todo)} remaining, fetching up to {args.n_peptides} this run")

    done, failed = [], []
    for name in todo:
        if len(done) >= args.n_peptides:
            print(f"reached --n_peptides={args.n_peptides}, stopping")
            break
        free_gb = _free_gb(args.sim_dir)
        if free_gb < args.min_free_gb:
            print(f"only {free_gb:.1f}GB free (< --min_free_gb={args.min_free_gb}), stopping early")
            break

        print(f"[{len(done) + len(failed) + 1}/{min(len(todo), args.n_peptides)}] {name} "
              f"({free_gb:.1f}GB free)...")
        ok, err = fetch_one(name, split_df, args.sim_dir, args.data_dir, args.cache_dir,
                             prefix, args.tps_eval_root, args.keep_raw)
        if ok:
            done.append(name)
            print(f"  OK")
        else:
            _cleanup_partial(name, args.sim_dir, args.data_dir, args.cache_dir, args.keep_raw)
            failed.append((name, err))
            print(f"  FAILED: {err}")

    print(f"\nDone: {len(done)} fetched, {len(failed)} failed, "
          f"{len(split_df) - len(todo) + len(done)}/{len(split_df)} total now available.")
    if failed:
        print("Failed peptides:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
