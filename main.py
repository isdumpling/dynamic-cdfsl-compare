from typing import List
import hydra
from omegaconf import DictConfig
try:
    from pytorch_lightning.callbacks import Callback
except ImportError:
    from pytorch_lightning.callbacks.base import Callback
import torch
import pytorch_lightning as pl
import argparse
from utils.custom_logger import CustomLogger
from data_loader.data_module import DataModule
from helper import load_system
from helper import config_init, refine_args
from helper.helper_slurm import run_cluster


def main(params: DictConfig, LightningSystem: pl.LightningModule, *args,
         **kwargs):
    params = config_init(params)
    params = refine_args(params)

    datamodule = DataModule(params.data)

    # Init PyTorch Lightning model ⚡
    model = LightningSystem(params, datamodule)

    if params.ckpt is not None and params.ckpt != 'none':
        if params.load_base:
            model.load_base(params.ckpt)
        else:
            ckpt = torch.load(
                params.ckpt,
                map_location=lambda storage, loc: storage)['state_dict']
            model.load_state_dict(ckpt, strict=not params.load_flexible)

    logger = CustomLogger(save_dir=params.logger.save_dir,
                          name=params.logger.name,
                          version=params.logger.version,
                          test=params.test,
                          disable_logfile=params.disable_logfile)

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = [
        hydra.utils.instantiate(callback_conf)
        for _, callback_conf in params["callbacks"].items()
    ] if "callbacks" in params else []

    # 兼容新版本PyTorch Lightning，直接使用字典参数初始化Trainer
    trainer_params = dict(params.trainer)
    trainer_params['logger'] = logger
    trainer_params['callbacks'] = callbacks
    if 'limit_val_batches' in trainer_params:
        trainer_params['limit_test_batches'] = trainer_params['limit_val_batches']
    
    # 移除新版本中不再支持的参数
    deprecated_params = ['checkpoint_callback', 'resume_from_checkpoint', 'weights_summary', 
                        'progress_bar_refresh_rate', 'process_position', 'flush_logs_every_n_steps',
                        'stochastic_weight_avg', 'terminate_on_nan', 'auto_scale_batch_size',
                        'auto_lr_find', 'replace_sampler_ddp', 'prepare_data_per_node',
                        'gpus', 'ipus', 'tpu_cores', 'num_processes', 'num_nodes',
                        'reload_dataloaders_every_epoch', 'weights_save_path', 'amp_backend',
                        'amp_level', 'distributed_backend', 'automatic_optimization',
                        'move_metrics_to_cpu', 'multiple_trainloader_mode', 'track_grad_norm']
    for param in deprecated_params:
        if param in trainer_params:
            del trainer_params[param]
    
    # 处理accelerator和devices参数（新版本的方式）
    if 'gpus' in params.trainer and params.trainer.gpus:
        trainer_params['accelerator'] = 'gpu'
        trainer_params['devices'] = params.trainer.gpus if isinstance(params.trainer.gpus, list) else params.trainer.gpus
    
    trainer = pl.Trainer(**trainer_params)

    if params.test:
        out = trainer.test(model)
        return out
    else:
        return trainer.fit(model)


@hydra.main(config_name="config", config_path="configs")
def hydra_main(cfg: DictConfig):
    lt_system = load_system(cfg.system_name)

    if cfg.launcher.name == "local":
        # add Lightning parse
        main(cfg, lt_system)
    elif cfg.launcher.name == "slurm":
        # submit job to slurm
        run_cluster(cfg, main, lt_system)
    elif cfg.launcher.name == "submitit_eval":
        from helper.helper_submitit_eval import submitit_eval_main
        submitit_eval_main(cfg, lt_system)


if __name__ == "__main__":
    hydra_main()
