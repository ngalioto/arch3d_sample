from arch3d.data.datamodule import HiC_DataModule
from arch3d.model.arch3d import ARCH3D, ARCH3D_Config
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import argparse
from pathlib import Path
from arch3d.conf.utils import load_config
import torch
import yaml
import os
from pytorch_lightning.utilities import rank_zero_only

# include when using Tensor Cores
torch.set_float32_matmul_precision('high')

import multiprocessing as mp
mp.set_start_method('spawn', force=True)


@rank_zero_only
def save_config_file(config: dict) -> None:

    save_dir = config['datamodule']['metadata_dir']
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, 'config.yaml')
        with open(fname, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

def main(
    config: dict
) -> None:
    
    if config['datamodule']['metadata_dir'] is not None:
        save_config_file(config)

    model_config = ARCH3D_Config(**config['model'])
    model = ARCH3D(model_config)

    
    datamodule = HiC_DataModule(**config['datamodule'], dataset_config=config['dataset'])

    if config['logger']['use_wandb']:
        del config['logger']['use_wandb']
        logger = WandbLogger(**config['logger'])
    else:
        logger = None
    
    trainer = L.Trainer(
        **config['trainer'], 
        logger=logger
    )

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config['ckpt_path']
    )
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pretrains the Hi-C foundation model.')

    parser.add_argument(
        '--config',
        type=Path,
        default='digitalcell/conf/pretrain.yaml',
        help='Path to the pretraining configuration file.'
    )

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)