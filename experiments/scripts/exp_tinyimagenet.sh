# Latent Replay
python -m experiments.train_incremental --multirun \
        experiment=latentreplay1_stinyimagenet n_classes=10  input_size=32 description="exp-3_tinyimagenet" \
        optimizer=Adam optim_params.lr=0.001 \
        strategy_params.buffer_update_mode="ClassBalanced" strategy_params.max_buffer_size=400 \
        model=ResNet18LatentReplay +latent_depth=1,2,3,4,5 \
        strategy_params.buffer_mb_size=64  strategy_params.train_epochs=100 \
        strategy_params.coef_exemplar_replay=0.0,1.0 \
        strategy_params.model_checkpoint_path="./ckpt_final.pt" \
        num_workers=5 end_after_n_exps=Null  wandb_proj=HyperCL save_results=True

# HyperResNet
python -m experiments.train_incremental --multirun \
        experiment=hyper-alg-reg-NM_stinyimagenet n_classes=10 input_size=32 description="exp-3_tinyimagenet" \
        model=HyperResNet18SH,HyperResNet18SPv1SH,HyperResNet18SPv2SH,HyperResNet18SPv3SH,HyperResNet18SPv4SH \
        model_params.embd_dim=32 model_params.hidden_size_1=50 \
        model_params.hidden_size_2=32 model_params.head_emb_dim=32 \
        optimizer=Adam optim_params.lr=0.001 \
        strategy_params.coef_hnet_replay=0.5 strategy_params.second_order=False \
        strategy_params.train_epochs=100 num_workers=5 \
        strategy_params.model_checkpoint_path="./ckpt_final.pt" \
        wandb_proj=HyperCL save_results=True

# Naive
python -m experiments.train_incremental --multirun \
        experiment=baseline_tinyimagenet n_classes=10  input_size=32 description="exp-3_tinyimagenet" \
        optimizer=Adam optim_params.lr=0.001 +model_params.freeze_depth=-1 \
        strategy=Naive strategy_params.train_epochs=100 model=ResNet18 multi_head=True \
        num_workers=5  seed=0 wandb_proj=HyperCL  save_results=True

# EWC
python -m experiments.train_incremental --multirun \
        experiment=ewc_tinyimagenet n_classes=10  input_size=32 description="exp-3_tinyimagenet" \
        optimizer=Adam optim_params.lr=0.001 +model_params.freeze_depth=-1 \
        strategy_params.train_epochs=100 strategy_params.ewc_lambda=1.0 model=ResNet18 multi_head=True \
        num_workers=5  seed=0,1,2 wandb_proj=HyperCL  save_results=True   
