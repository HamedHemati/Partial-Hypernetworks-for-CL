# Latent Replay
python -m experiments.train_incremental --multirun \
        experiment=latentreplay1_noisycifar description="exp-4_noisycifar" \
        optimizer=Adam optim_params.lr=0.001 \
        bnch_params.n_experiences=4  bnch_params.n_exp_cls=5  bnch_params.add_noise=True,False \
        strategy_params.buffer_update_mode="ClassBalanced" strategy_params.max_buffer_size=200 \
        model=ResNet18LatentReplay +latent_depth=1,2,3,4,5 \
        strategy_params.buffer_mb_size=64  strategy_params.train_epochs=100 \
        strategy_params.coef_exemplar_replay=1.0 \
        strategy_params.model_checkpoint_path="./checkpoints/ckpt_final.pt" \
        wandb_proj=HyperCL save_results=True

# HyperResNet
python -m experiments.train_incremental --multirun \
        experiment=hyper-alg-reg-NM_noisycifar n_classes=5 input_size=32 description="exp-4_noisycifar" \
        bnch_params.n_experiences=4  bnch_params.n_exp_cls=5  bnch_params.add_noise=True,False \
        model=HyperResNet18SPv3SH,HyperResNet18SPv4SH,HyperResNet18SPv2SH \
        model_params.embd_dim=32 model_params.hidden_size_1=50 \
        model_params.hidden_size_2=32 model_params.head_emb_dim=32 \
        optimizer=Adam optim_params.lr=0.001 \
        strategy_params.coef_hnet_replay=0.5 strategy_params.second_order=False \
        strategy_params.train_epochs=100 num_workers=5 \
        strategy_params.model_checkpoint_path="./checkpoints/ckpt_final.pt" \
        wandb_proj=HyperCL save_results=True
