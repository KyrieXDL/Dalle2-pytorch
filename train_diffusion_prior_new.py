import torch
import torch.utils.data
from torch.utils.data import DataLoader
from models.dalle2_prior import DiffusionPriorNetwork, DiffusionPrior
from trainer.dalle2_prior_trainer import DiffusionPriorTrainer
from accelerate.utils import set_seed
from accelerate import Accelerator
from models.clip_model import OpenAIClipAdapter
from torchvision import transforms
from tqdm import tqdm
import os
from mydataset.dalle2_dataset import Dalle2Dataset, collate_func
from configs.dalle2_configs import TrainDiffusionPriorConfig
import click
OUTPUT_DIR = './output'


def train_prior(diff_trainer, train_dataloader, device):
    batch_size = 1
    train_size = len(train_dataloader)
    idx_list = range(0, train_size, batch_size)
    total_loss = 0
    step = 0

    for batch in tqdm(train_dataloader):
        image, text = batch
        # print(text.shape, image.shape)
        loss = diff_trainer(text, image)
        diff_trainer.accelerator.backward(loss)
        total_loss += loss.item()
        diff_trainer.update()

        if step % 100 == 0 and diff_trainer.accelerator.is_main_process:
            save_path = os.path.join(OUTPUT_DIR, 'prior', 'model.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(diff_trainer.accelerator.unwrap_model(diff_trainer.diffusion_prior).state_dict(), save_path)
            print(f"average loss: {total_loss / (step + 1)}")

        step += 1


def main(config):
    # init
    accelerator = Accelerator()
    set_seed(config.train.random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = accelerator.device

    # build model
    clip = OpenAIClipAdapter(config.prior.clip.model).to(device)
    prior_network = DiffusionPriorNetwork(**config.prior.net.dict()).to(device)
    prior_kwargs = config.prior.dict()
    prior_kwargs.pop('clip')
    prior_kwargs.pop('net')
    diffusion_prior = DiffusionPrior(net=prior_network, clip=clip, **prior_kwargs).to(device)

    train_kwargs = config.train.dict()
    num_epochs = train_kwargs.pop('epochs')
    diff_trainer = DiffusionPriorTrainer(diffusion_prior,
                                         accelerator=accelerator,
                                         ema_beta=0.99,
                                         ema_update_after_step=1000,
                                         ema_update_every=10,
                                         )

    # build dataloader
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    dataset = Dalle2Dataset(config.train.train_img_path, config.train.train_annot_path, transform)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=16,
                            collate_fn=collate_func, drop_last=True)
    dataloader = accelerator.prepare(dataloader)

    for epoch in range(num_epochs):
        train_prior(diff_trainer, dataloader, device)


if __name__ == '__main__':
    config = TrainDiffusionPriorConfig.from_json_path('./configs/train_prior_config.json')

    print(config)
    main(config)