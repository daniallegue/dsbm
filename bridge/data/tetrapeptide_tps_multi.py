"""Amortized, multi-peptide version of TetrapeptideTPSDataset (bridge/data/tetrapeptide_tps.py):
instead of a single peptide's empirical (frames, torsions) marginal, each dataset item is drawn
from a peptide chosen by a fixed schedule, and returns a conditioning vector `y` -- the peptide's
per-residue amino-acid-type indices -- alongside the ambient state vector `x`. The drift network
(bridge/models/basic/protein_cond.py::ScoreNetworkProteinCond) embeds `y` and adds it to its
invariant hidden features, letting one network amortize across every peptide in the schedule
(bridge/runners/config_getters.py wires this up with `cdsb: True`).

PAIRING CONTRACT (read before touching `peptide_seed` or dataset construction order):
bridge/trainer_dbdsb.py pairs x0 = start_dataset[i] with x1 = end_dataset[i] purely by index --
that's the existing "independent coupling" scheme, and it is only correct here if index i refers
to the SAME peptide in both the start-endpoint dataset and the end-endpoint dataset (otherwise
we'd bridge one peptide's start state to a different peptide's end state). This class makes that
true by deriving `peptide_schedule` from `peptide_seed` alone (not from `endpoint`); the caller
MUST construct the start and end datasets with the same `peptide_seed` (only `sample_seed`, which
picks the raw MD frame within a peptide's state, should differ -- mirroring how the single-peptide
TetrapeptideTPSDataset already uses seed=1 for start / seed=2 for end). The `paired = True` class
attribute tells bridge/trainer_dbdsb.py::build_dataloader to force shuffle=False on any DataLoader
built over this dataset, so batch position and dataset index coincide forever (see that function's
comment) -- without that, two independently-shuffled DataLoaders would desync the index alignment
this class sets up.
"""
import os

import numpy as np
import torch

from .tetrapeptide_tps import _VENDOR_MDGEN, _seqres_tensor, atom14_frame_to_ambient


def _available_peptides(split, data_dir, cache_dir, max_peptides=None):
    """Peptide names from vendor/mdgen/splits/{split}.csv that have both raw MD data
    ({data_dir}/{peptide}.npy) and a precomputed state-pair cache
    ({cache_dir}/{peptide}/{peptide}_states.npz) on disk -- see TetrapeptideTPSDataset's
    docstring for why the latter must be precomputed offline (needs pyemma)."""
    import pandas as pd
    split_path = os.path.join(_VENDOR_MDGEN, "splits", f"{split}.csv")
    names = pd.read_csv(split_path, index_col="name").index.tolist()

    available = [
        name for name in names
        if os.path.isfile(os.path.join(data_dir, f"{name}.npy"))
        and os.path.isfile(os.path.join(cache_dir, name, f"{name}_states.npz"))
    ]
    print(f"[tetrapeptide_tps_multi] {len(available)}/{len(names)} peptides from '{split}' "
          f"available on disk (data_dir={data_dir}, cache_dir={cache_dir})")
    if max_peptides is not None:
        available = available[:max_peptides]

    if len(available) == 0:
        raise FileNotFoundError(
            f"no peptide in {split_path} has both '{{data_dir}}/{{peptide}}.npy' and "
            f"'{{cache_dir}}/{{peptide}}/{{peptide}}_states.npz' -- run "
            f"vendor/mdgen/scripts/prep_sims.py and scripts/precompute_tps_states.py "
            f"(under the `tps_eval` conda env) for at least one peptide first. "
            f"data_dir={data_dir}, cache_dir={cache_dir}")
    return available


class TetrapeptideTPSMultiDataset(torch.utils.data.Dataset):
    """Empirical (frames, torsions) marginal for one endpoint ('start' or 'end'), amortized
    over many tetrapeptides. See module docstring for the index-pairing contract this class
    relies on."""

    paired = True  # bridge/trainer_dbdsb.py::build_dataloader forces shuffle=False for this

    def __init__(self, endpoint, split, data_dir, cache_dir, n_samples, peptide_seed=0,
                 sample_seed=1, max_peptides=None):
        assert endpoint in ("start", "end")
        self.endpoint = endpoint
        self.data_dir = data_dir
        self.cache_dir = cache_dir

        peptide_list = _available_peptides(split, data_dir, cache_dir, max_peptides=max_peptides)
        self.peptide_schedule = np.random.default_rng(peptide_seed).choice(
            peptide_list, size=n_samples, replace=True)

        self._states_cache = {}  # peptide -> (idxs, seqres_tensor, L)
        self._mmap_cache = {}    # peptide -> memmap array

        rng_sample = np.random.default_rng(sample_seed)
        self.drawn_idx = np.empty(n_samples, dtype=np.int64)
        for i, peptide in enumerate(self.peptide_schedule):
            idxs, _, _ = self._get_states(peptide)
            self.drawn_idx[i] = rng_sample.choice(idxs)

    def _get_states(self, peptide):
        cached = self._states_cache.get(peptide)
        if cached is not None:
            return cached

        states_path = os.path.join(self.cache_dir, peptide, f"{peptide}_states.npz")
        states = np.load(states_path)
        idxs = states["start_idxs"] if self.endpoint == "start" else states["end_idxs"]

        import pandas as pd
        splits_df = pd.read_csv(os.path.join(_VENDOR_MDGEN, "splits", "4AA.csv"), index_col="name")
        seqres_tensor = _seqres_tensor(splits_df.seqres[peptide])
        L = len(splits_df.seqres[peptide])

        result = (idxs, seqres_tensor, L)
        self._states_cache[peptide] = result
        return result

    def _get_mmap(self, peptide):
        arr = self._mmap_cache.get(peptide)
        if arr is None:
            arr = np.lib.format.open_memmap(os.path.join(self.data_dir, f"{peptide}.npy"), mode="r")
            self._mmap_cache[peptide] = arr
        return arr

    def __len__(self):
        return len(self.peptide_schedule)

    def __getitem__(self, index):
        peptide = str(self.peptide_schedule[index])
        _, seqres_tensor, _ = self._get_states(peptide)
        arr = self._get_mmap(peptide)

        frame = np.copy(arr[self.drawn_idx[index]]).astype(np.float32)  # (L, 14, 3)
        x = atom14_frame_to_ambient(frame, seqres_tensor)

        # Amino-acid-type indices as float: DBDSB_CacheLoader round-trips `y` through a
        # float32 memmap cache (bridge/data/cacheloader.py), and small integers survive that
        # round trip exactly. ScoreNetworkProteinCond casts back to .long() before embedding.
        y = seqres_tensor.float()
        return x, y
