"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger,evaluate_policy

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "flow"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-2)
    train_losses = []
    eval_rewards = []
    ### TODO: PUT YOUR MAIN TRAINING LOOP HERE ###
    evaluate_counter = 0
    for epoch in range(config.num_epochs):
        for states,actions in loader:
            states = states.to(device)
            actions = actions.to(device)
            loss = model.compute_loss(states,actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            evaluate_counter += 1
            if evaluate_counter  % config.log_interval == 0:
                train_losses.append(loss.item())
                wandb.log({"train/loss": loss.item()}, step=evaluate_counter)          
            if evaluate_counter % config.eval_interval == 0:
                # print('\n' + '='*50)
                # print(f"Action Debug: value = {actions}")
                # print(f"Action Debug: type = {type(actions)}")
                # if isinstance(actions, torch.Tensor):
                    # print(f"Action Debug: device = {actions.device}, shape = {actions.shape}")
                    # print("="*50 + '\n')
                    # if torch.is_tensor(actions):
                    #     actions = actions.detach().cpu().numpy()
                evaluate_policy(model, normalizer, device, model.chunk_size, config.video_size, 
                                config.num_video_episodes, config.flow_num_steps, evaluate_counter, logger)
                if len(logger.rows) > 0:
                    last_log = logger.rows[-1]
                    if "eval/mean_reward" in last_log:
                        reward = last_log["eval/mean_reward"]
                        eval_rewards.append(reward)
                    #wandb.log({"eval/mean_reward": eval_results['reward']}, step=evaluate_counter)
    fig = plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time for %s Policy' % config.policy_type.upper())
    plt.legend()
    plt.grid()
    plt.savefig(log_dir / "train_loss.png")
    plt.close(fig)
    fig = plt.figure()
    plt.plot(range(len(eval_rewards)), eval_rewards, label='Eval Reward')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Reward')
    plt.title('Evaluation Reward over Time for %s Policy' % config.policy_type.upper())
    plt.legend()
    plt.grid()
    plt.savefig(log_dir / "eval_reward.png")
    plt.close(fig)

    logger.dump_for_grading()
def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
