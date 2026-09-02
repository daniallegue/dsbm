import torch
from omegaconf import OmegaConf
import hydra
from ..models import *
from .plotters import *
import torchvision.datasets
import torchvision.transforms as transforms
import os
from functools import partial
from .logger import CSVLogger, WandbLogger, Logger
from torch.utils.data import DataLoader
from bridge.data.afhq import AFHQ
from bridge.data.downscaler import DownscalerDataset
from bridge.data.spherical_harmonics import SphericalHarmonicDataset
from bridge.data.torus import TorusMixtureDataset
from bridge.data.alanine_dipeptide import AlanineDipeptideFramesDataset
from bridge.data.tetrapeptide_tps import TetrapeptideTPSDataset
from bridge.data.tetrapeptide_tps_multi import TetrapeptideTPSMultiDataset

cmp = lambda x: transforms.Compose([*x])

def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)


def get_plotter(runner, args):
    dataset_tag = getattr(args, DATASET)
    if dataset_tag in [DATASET_MNIST, DATASET_EMNIST, DATASET_CIFAR10] or dataset_tag.startswith(DATASET_AFHQ):
        return ImPlotter(runner, args)
    elif dataset_tag in [DATASET_DOWNSCALER_LOW, DATASET_DOWNSCALER_HIGH]:
        return DownscalerPlotter(runner, args)
    else:
        return Plotter(runner, args)


# Model
# --------------------------------------------------------------------------------

MODEL = 'Model'
BASIC_MODEL = 'Basic'
BASIC_PROTEIN_COND_MODEL = 'BasicProteinCond'
UNET_MODEL = 'UNET'
DOWNSCALER_UNET_MODEL = 'DownscalerUNET'
DDPMPP_MODEL = 'DDPMpp'

NAPPROX = 2000


def get_model(args):
    model_tag = getattr(args, MODEL)

    if model_tag == UNET_MODEL:
        image_size = args.data.image_size

        if args.model.channel_mult is not None:
            channel_mult = args.model.channel_mult
        else:
            if image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 160:
                channel_mult = (1, 2, 2, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 32:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 28:
                channel_mult = (0.5, 1, 1)
            else:
                raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = []
        for res in args.model.attention_resolutions.split(","):
            if image_size % int(res) == 0:
                attention_ds.append(image_size // int(res))

        kwargs = {
            "in_channels": args.data.channels,
            "model_channels": args.model.num_channels,
            "out_channels": args.data.channels,
            "num_res_blocks": args.model.num_res_blocks,
            "attention_resolutions": tuple(attention_ds),
            "dropout": args.model.dropout,
            "channel_mult": channel_mult,
            "num_classes": None,
            "use_checkpoint": args.model.use_checkpoint,
            "num_heads": args.model.num_heads,
            "use_scale_shift_norm": args.model.use_scale_shift_norm,
            "resblock_updown": args.model.resblock_updown,
            "temb_scale": args.model.temb_scale
        }

        net = UNetModel(**kwargs)

    elif model_tag == DOWNSCALER_UNET_MODEL:
        image_size = args.data.image_size
        channel_mult = args.model.channel_mult

        kwargs = {
            "in_channels": args.data.channels,
            "cond_channels": args.data.cond_channels, 
            "model_channels": args.model.num_channels,
            "out_channels": args.data.channels,
            "num_res_blocks": args.model.num_res_blocks,
            "dropout": args.model.dropout,
            "channel_mult": channel_mult,
            "temb_scale": args.model.temb_scale, 
            "mean_bypass": args.model.mean_bypass,
            "scale_mean_bypass": args.model.scale_mean_bypass,
            "shift_input": args.model.shift_input,
            "shift_output": args.model.shift_output,
        }

        net = DownscalerUNetModel(**kwargs)

    elif model_tag == DDPMPP_MODEL:
        # assert args.data.image_size == 512
        class Config():
            pass
        config = Config()
        config.model = Config()
        config.data = Config()
        config.model.scale_by_sigma = args.model.scale_by_sigma
        config.model.normalization = args.model.normalization
        config.model.nonlinearity = args.model.nonlinearity
        config.model.nf = args.model.nf
        config.model.ch_mult = args.model.ch_mult
        config.model.num_res_blocks = args.model.num_res_blocks
        config.model.attn_resolutions = args.model.attn_resolutions
        config.model.dropout = args.model.dropout
        config.model.resamp_with_conv = args.model.resamp_with_conv
        config.model.conditional = args.model.conditional
        config.model.fir = args.model.fir
        config.model.fir_kernel = args.model.fir_kernel
        config.model.skip_rescale = args.model.skip_rescale
        config.model.resblock_type = args.model.resblock_type
        config.model.progressive = args.model.progressive
        config.model.progressive_input = args.model.progressive_input
        config.model.progressive_combine = args.model.progressive_combine
        config.model.attention_type = args.model.attention_type
        config.model.init_scale = args.model.init_scale
        config.model.fourier_scale = args.model.fourier_scale
        config.model.conv_size = args.model.conv_size
        config.model.embedding_type = args.model.embedding_type

        config.data.image_size = args.data.image_size
        config.data.num_channels = args.data.channels
        config.data.centered = True # assumes data is within -1, 1 and so the model will do no adjustments to it


        net = NCSNpp(config)

    elif model_tag == BASIC_MODEL:
        kwargs = {
            "encoder_layers": args.model.encoder_layers,
            "temb_dim": args.model.temb_dim,
            "decoder_layers": args.model.decoder_layers,
            "x_dim": args.data.dim,
            "temb_max_period": args.model.temb_max_period,
        }

        net = ScoreNetwork(**kwargs)

    elif model_tag == BASIC_PROTEIN_COND_MODEL:
        kwargs = {
            "encoder_layers": args.model.encoder_layers,
            "temb_dim": args.model.temb_dim,
            "decoder_layers": args.model.decoder_layers,
            "x_dim": args.data.dim,
            "n_residues": args.data.n_residues,
            "cond_emb_dim": args.model.cond_emb_dim,
            "temb_max_period": args.model.temb_max_period,
        }

        net = ScoreNetworkProteinCond(**kwargs)

    return net

# Optimizer
# --------------------------------------------------------------------------------

def get_optimizer(net, args):
    lr = args.lr
    optimizer = args.optimizer
    if optimizer == 'Adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=args.weight_decay)


# Dataset
# --------------------------------------------------------------------------------

DATASET = 'Dataset'
DATASET_TRANSFER = 'Dataset_transfer'
DATASET_MNIST = 'mnist'
DATASET_EMNIST = 'emnist'
DATASET_CIFAR10 = 'cifar10'
DATASET_AFHQ = 'afhq'
DATASET_DOWNSCALER_LOW = 'downscaler_low'
DATASET_DOWNSCALER_HIGH = 'downscaler_high'
DATASET_SPHERICAL_HARMONICS = 'spherical_harmonics'
DATASET_TORUS_GAUSSIANS = 'torus_gaussians'
DATASET_ALANINE_DIPEPTIDE = 'alanine_dipeptide'
DATASET_TETRAPEPTIDE_TPS = 'tetrapeptide_tps'
DATASET_TETRAPEPTIDE_TPS_MULTI = 'tetrapeptide_tps_multi'

def get_datasets(args):
    dataset_tag = getattr(args, DATASET)

    # INITIAL (DATA) DATASET

    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)

    # MNIST DATASET
    if dataset_tag == DATASET_MNIST:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'mnist')
        load = args.load
        assert args.data.channels == 1
        assert args.data.image_size == 28
        train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        init_ds = torchvision.datasets.MNIST(root=root, train=True, transform=cmp(train_transform), download=True)

    # CIFAR10 DATASET
    if dataset_tag == DATASET_CIFAR10:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'cifar10')
        load = args.load
        assert args.data.channels == 3
        assert args.data.image_size == 32
        train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        if args.data.random_flip:
            train_transform.insert(0, transforms.RandomHorizontalFlip())
        
        init_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=cmp(train_transform), download=True)
        
    # AFHQ DATASET
    if dataset_tag.startswith(DATASET_AFHQ):
        assert args.data.image_size == 512
        animal_type = dataset_tag.split('_')[1]
        init_ds = AFHQ(root_dir=os.path.join(args.paths.afhq_path, 'train'), animal_type=animal_type)

    # Downscaler dataset
    if dataset_tag == DATASET_DOWNSCALER_HIGH:
        root = os.path.join(data_dir, 'downscaler')
        train_transform = [transforms.Normalize((0.,), (1.,))]
        assert not args.data.random_flip
        # if args.data.random_flip:
        #     train_transform = train_transform + [
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.RandomVerticalFlip(p=0.5),
        #         transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        #     ]
        wavenumber = args.data.get('wavenumber', 0)
        split = args.data.get('split', "train")
        
        init_ds = DownscalerDataset(root=root, resolution=512, wavenumber=wavenumber, split=split, transform=cmp(train_transform))

    # Spherical harmonics dataset
    if dataset_tag == DATASET_SPHERICAL_HARMONICS:
        assert args.data.dim == 3
        init_ds = SphericalHarmonicDataset(l=args.data.l_init, m=args.data.m_init, n_samples=args.data.n_samples)

    # Torus (flat, zero curvature) mixture dataset
    if dataset_tag == DATASET_TORUS_GAUSSIANS:
        assert args.data.dim == 2
        init_ds = TorusMixtureDataset(n_modes=args.data.n_modes_init, kappa=args.data.kappa,
                                       n_samples=args.data.n_samples, ring_radius=args.data.ring_radius, seed=1)

    # Alanine dipeptide relative-backbone-frame dataset (SO(3)), source basin
    if dataset_tag == DATASET_ALANINE_DIPEPTIDE:
        assert args.data.dim == 9
        init_ds = AlanineDipeptideFramesDataset(
            basin=args.data.basin_init, n_samples=args.data.n_samples,
            cache_dir=os.path.join(data_dir, 'alanine_dipeptide'),
            temperature_k=args.data.md_temperature_k, friction_per_ps=args.data.md_friction_per_ps,
            save_every=args.data.md_save_every, ball_radius=args.data.basin_ball_radius, seed=1)

    # Tetrapeptide TPS SE(3)^(L-1) x T^(n_torsions*L) dataset (root-relative frames,
    # see bridge/data/tetrapeptide_tps.py::root_relative_frames), start-state endpoint
    if dataset_tag == DATASET_TETRAPEPTIDE_TPS:
        assert args.data.dim == 3 * (args.data.n_residues - 1) + 9 * (args.data.n_residues - 1) + args.data.n_torsions * args.data.n_residues
        init_ds = TetrapeptideTPSDataset(
            peptide=args.data.peptide, endpoint="start",
            data_dir=hydra.utils.to_absolute_path(args.data.data_dir),
            cache_dir=hydra.utils.to_absolute_path(args.data.cache_dir),
            n_samples=args.data.n_samples, seed=1)

    # Amortized (peptide-conditioned) tetrapeptide TPS dataset: same manifold/dim as above, but
    # each item is drawn from one of many peptides (bridge/data/tetrapeptide_tps_multi.py), with
    # `y` carrying that peptide's per-residue amino-acid-type indices for the conditioned network
    # (bridge/models/basic/protein_cond.py). start-endpoint here; end-endpoint in get_final_dataset.
    if dataset_tag == DATASET_TETRAPEPTIDE_TPS_MULTI:
        assert args.data.dim == 3 * (args.data.n_residues - 1) + 9 * (args.data.n_residues - 1) + args.data.n_torsions * args.data.n_residues
        init_ds = TetrapeptideTPSMultiDataset(
            endpoint="start", split=args.data.split,
            data_dir=hydra.utils.to_absolute_path(args.data.data_dir),
            cache_dir=hydra.utils.to_absolute_path(args.data.cache_dir),
            n_samples=args.data.n_samples, peptide_seed=args.data.peptide_schedule_seed,
            sample_seed=1, max_peptides=args.data.get("max_peptides", None))

    # FINAL DATASET

    final_ds, mean_final, var_final = get_final_dataset(args, init_ds)
    return init_ds, final_ds, mean_final, var_final


def get_final_dataset(args, init_ds):
    if args.transfer:
        data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)
        dataset_transfer_tag = getattr(args, DATASET_TRANSFER)
        mean_final = torch.tensor(0.)
        var_final = torch.tensor(1.*10**3)  # infty like

        if dataset_transfer_tag == DATASET_EMNIST:
            from ..data.emnist import FiveClassEMNIST
            # data_tag = args.data.dataset
            root = os.path.join(data_dir, 'emnist')
            load = args.load
            assert args.data.channels == 1
            assert args.data.image_size == 28
            train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            final_ds = FiveClassEMNIST(root=root, train=True, download=True, transform=cmp(train_transform))

        # AFHQ DATASET
        if dataset_transfer_tag.startswith(DATASET_AFHQ):
            assert args.data.image_size == 512
            animal_type = dataset_transfer_tag.split('_')[1]
            final_ds = AFHQ(root_dir=os.path.join(args.paths.afhq_path, 'train'), animal_type=animal_type)

        if dataset_transfer_tag == DATASET_DOWNSCALER_LOW:
            root = os.path.join(data_dir, 'downscaler')
            train_transform = [transforms.Normalize((0.,), (1.,))]
            if args.data.random_flip:
                train_transform = train_transform + [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                ]

            split = args.data.get('split', "train")
            
            final_ds = DownscalerDataset(root=root, resolution=64, split=split, transform=cmp(train_transform))

        if dataset_transfer_tag == DATASET_SPHERICAL_HARMONICS:
            assert args.data.dim == 3
            final_ds = SphericalHarmonicDataset(l=args.data.l_final, m=args.data.m_final, n_samples=args.data.n_samples)

        if dataset_transfer_tag == DATASET_TORUS_GAUSSIANS:
            assert args.data.dim == 2
            phase_offset = np.pi / args.data.n_modes_final
            final_ds = TorusMixtureDataset(n_modes=args.data.n_modes_final, kappa=args.data.kappa,
                                            n_samples=args.data.n_samples, ring_radius=args.data.ring_radius,
                                            phase_offset=phase_offset, seed=2)

        if dataset_transfer_tag == DATASET_ALANINE_DIPEPTIDE:
            assert args.data.dim == 9
            final_ds = AlanineDipeptideFramesDataset(
                basin=args.data.basin_final, n_samples=args.data.n_samples,
                cache_dir=os.path.join(data_dir, 'alanine_dipeptide'),
                temperature_k=args.data.md_temperature_k, friction_per_ps=args.data.md_friction_per_ps,
                save_every=args.data.md_save_every, ball_radius=args.data.basin_ball_radius, seed=2)

        if dataset_transfer_tag == DATASET_TETRAPEPTIDE_TPS:
            assert args.data.dim == 3 * (args.data.n_residues - 1) + 9 * (args.data.n_residues - 1) + args.data.n_torsions * args.data.n_residues
            final_ds = TetrapeptideTPSDataset(
                peptide=args.data.peptide, endpoint="end",
                data_dir=hydra.utils.to_absolute_path(args.data.data_dir),
                cache_dir=hydra.utils.to_absolute_path(args.data.cache_dir),
                n_samples=args.data.n_samples, seed=2)

        if dataset_transfer_tag == DATASET_TETRAPEPTIDE_TPS_MULTI:
            assert args.data.dim == 3 * (args.data.n_residues - 1) + 9 * (args.data.n_residues - 1) + args.data.n_torsions * args.data.n_residues
            # peptide_seed MUST match get_datasets' init_ds call above -- see
            # bridge/data/tetrapeptide_tps_multi.py's pairing contract docstring; only
            # sample_seed (which raw MD frame within the peptide's state) differs.
            final_ds = TetrapeptideTPSMultiDataset(
                endpoint="end", split=args.data.split,
                data_dir=hydra.utils.to_absolute_path(args.data.data_dir),
                cache_dir=hydra.utils.to_absolute_path(args.data.cache_dir),
                n_samples=args.data.n_samples, peptide_seed=args.data.peptide_schedule_seed,
                sample_seed=2, max_peptides=args.data.get("max_peptides", None))

    else:
        if args.adaptive_mean:
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, num_workers=args.num_workers, worker_init_fn=worker_init_fn)))[0]
            mean_final = vec.mean(axis=0)
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
        elif args.final_adaptive:
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, num_workers=args.num_workers, worker_init_fn=worker_init_fn)))[0]
            mean_final = vec.mean(axis=0)
            var_final = vec.var(axis=0)
        else:
            mean_final = eval(args.mean_final) if isinstance(args.mean_final, str) else torch.tensor([args.mean_final])
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
        final_ds = None

    return final_ds, mean_final, var_final


def get_valid_test_datasets(args):
    valid_ds, test_ds = None, None

    dataset_tag = getattr(args, DATASET)
    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)

    # MNIST DATASET
    if dataset_tag == DATASET_MNIST:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'mnist')
        load = args.load
        assert args.data.channels == 1
        assert args.data.image_size == 28
        test_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        valid_ds = None
        test_ds = torchvision.datasets.MNIST(root=root, train=False, transform=cmp(test_transform), download=True)
    
    # # CIFAR10 DATASET
    # if dataset_tag == DATASET_CIFAR10:
    #     # data_tag = args.data.dataset
    #     root = os.path.join(data_dir, 'cifar10')
    #     load = args.load
    #     assert args.data.channels == 3
    #     assert args.data.image_size == 32
    #     test_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    #     valid_ds = None
    #     test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=cmp(test_transform), download=True)

    return valid_ds, test_ds


# Logger
# --------------------------------------------------------------------------------

LOGGER = 'LOGGER'
CSV_TAG = 'CSV'
WANDB_TAG = 'Wandb'
NOLOG_TAG = 'NONE'


def get_logger(args, name):
    logger_tag = getattr(args, LOGGER)

    if logger_tag == CSV_TAG:
        kwargs = {'save_dir': args.CSV_log_dir, 'name': name, 'flush_logs_every_n_steps': 1}
        return CSVLogger(**kwargs)

    if logger_tag == WANDB_TAG:
        log_dir = os.getcwd()
        if not args.use_default_wandb_name:
            run_name = os.path.normpath(os.path.relpath(log_dir, os.path.join(
                hydra.utils.to_absolute_path(args.paths.experiments_dir_name), args.name))).replace("\\", "/")
        else:
            run_name = None
        data_tag = args.data.dataset
        config = OmegaConf.to_container(args, resolve=True)

        wandb_entity = os.environ['WANDB_ENTITY']
        assert len(wandb_entity) > 0, "WANDB_ENTITY not set"

        kwargs = {'name': run_name, 'project': 'dsbm_' + args.name, 'prefix': name, 'entity': wandb_entity,
                  'tags': [data_tag], 'config': config, 'id': str(args.wandb_id) if args.wandb_id is not None else None}
        return WandbLogger(**kwargs)

    if logger_tag == NOLOG_TAG:
        return Logger()
