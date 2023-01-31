import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import sys
import time
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from models import CentralModel
from utils import num_to_groups
sys.path.insert(1, '../datasets/')
from samplers import sample
from losses import p_losses
from dataset_wraper import DatasetWraper
from schedulers import Scheduler

def train(data, model, optimizer, scheduler, cfg, device):
    model.train()
    for epoch in range(cfg.training.epochs):
        experiment_name = f"{data}_{model}_{cfg.optimizer.type}_{scheduler}_{cfg.training.loss}__ep{epoch}"
        print(f'conducting experiment {experiment_name}')
        epoch_loss = []
        for step, batch in enumerate(data.DataLoader()):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            t = torch.randint(0, cfg.scheduler.timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, scheduler, batch, t, loss_type=cfg.training.loss)
            epoch_loss.append(loss.item())

            if step % 30 == 0:
                print("loss:", sum(epoch_loss) / len(epoch_loss), 'at step', step, ' out of ', len(data.DataLoader()))

            loss.backward()
            optimizer.step()

        print(f'Loss after epoch {epoch}: {sum(epoch_loss) / len(epoch_loss)}')
        torch.save(model.state_dict(), f'../weights/{experiment_name}.pkl')

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    device = cfg.device
    data = DatasetWraper(cfg)
    scheduler = Scheduler(cfg.scheduler.type, timesteps=cfg.scheduler.timesteps)
    model = CentralModel(cfg).to(device)
    optimizer_class = hydra.utils.get_class(cfg.optimizer._target_)
    optimizer = optimizer_class(model.parameters(), lr=cfg.optimizer.lr)
    train(data, model, optimizer, scheduler, cfg, device)


if __name__ == "__main__":
    main()
