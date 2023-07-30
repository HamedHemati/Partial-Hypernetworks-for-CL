import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_pl_callbacks(config):
    # Trainer
    callbacks = []
    if config.save_results:
        checkpoint_callback = ModelCheckpoint(monitor=config.ckpt_save_metric, mode=config.ckpt_save_mode,
                                              dirpath=config.paths["checkpoints"],
                                              filename="ckpt-best-{" + config.ckpt_save_metric + ":.2f}")
        callbacks.append(checkpoint_callback)

    return callbacks


def get_pl_logger(config):
    logger = True
    if config.log_to_wandb:
        save_dir = os.path.join(config.outputs_dir, "temp")
        if config.save_results:
            save_dir = config.paths["results"]
        
        wandb_logger = WandbLogger(project=config.wandb_proj, name=config.exp_name, id=config.exp_name,
                                   save_dir=save_dir)
        logger = wandb_logger

    return logger
