import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import sys
# from torch.optim import Adam
# from torchvision.utils import save_image
# from torch.utils.data import DataLoader, random_split
from models import CentralModel
sys.path.insert(1, '../datasets/')
# from samplers import sample
from losses import p_losses
from dataset_wraper import DatasetWraper
from schedulers import Scheduler
# import os
import time
import wandb


def evaluate (data, model, scheduler, device):
    model.eval()
    total_loss = []
    start_eval_time = time.time()
    
    for step, batch in enumerate(data.TestDataloader()):
        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)
        t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()
        loss = p_losses(model, scheduler, batch, t, loss_type="huber")
        total_loss += [loss.item()]

    mean_loss = sum(total_loss) / len(total_loss)
    print(f'eval loss {mean_loss:.4f}, with time {time.time() - start_eval_time:.2f}s')
    return mean_loss

def train(data, model, optimizer, scheduler, cfg, device):
    model.train()
    formatted_time = time.strftime('%m%d-%H%M')
    for epoch in range(cfg.training.epochs):
        start_epoch_time = time.time()
        experiment_name = f"{data}_{model}_{cfg.optimizer.type}_{scheduler}_{cfg.training.loss}_{formatted_time}"
        print(f'conducting experiment {experiment_name}')

        epoch_loss = []
        for step, batch in enumerate(data.DataLoader()):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            t = torch.randint(0, cfg.scheduler.timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, scheduler, batch, t, loss_type=cfg.training.loss)
            epoch_loss.append(loss.item())

            if step % 100 == 0:
                print(f'loss: {loss:.4f} at step {step} out of {len(data.DataLoader())}')

            loss.backward()
            optimizer.step()

        mean_loss = sum(epoch_loss) / len(epoch_loss)
        print(f'Epoch {epoch}: loss={mean_loss:.4f}, with time {time.time() - start_epoch_time:.2f}s')
        wandb.log({'training loss': mean_loss, 'epoch': epoch})

        if epoch % cfg.training.eval_every == 0:
            eval_loss = evaluate(data, model, scheduler, device)
            wandb.log({'evaluation loss': eval_loss, 'epoch': epoch})
        if epoch % cfg.training.save_every == 0:
            torch.save(model.state_dict(), f'../weights/{experiment_name}__{formatted_time}.pkl')
        if epoch % cfg.training.visualize_every == 0:
            data.visualize(model, scheduler, experiment_name, epoch, mean_loss)

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    wandb.init(config=cfg,
               project='DEQ+Diffusion toy experiments',
               entity='luke_and_xingjian',
               notes='one toy experiment',
               name=f'{cfg.dataset.name}_{cfg.model.type}_{cfg.optimizer.type}_{cfg.scheduler.type}_{cfg.training.loss}',
               job_type="training",
               reinit=True)
    print(OmegaConf.to_yaml(cfg))
    device = cfg.device
    data = DatasetWraper(cfg)
    scheduler = Scheduler(cfg.scheduler.type, timesteps=cfg.scheduler.timesteps)
    model = CentralModel(cfg).to(device)
    optimizer_class = hydra.utils.get_class(cfg.optimizer._target_)
    optimizer = optimizer_class(model.parameters(), lr=cfg.optimizer.lr)
    train(data, model, optimizer, scheduler, cfg, device)


if __name__ == "__main__":
    main()
