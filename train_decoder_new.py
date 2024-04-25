import torch
import torch.utils
from torch.utils.data import DataLoader, IterableDataset, BatchSampler
from models.unet import Unet
from models.dalle2_decoder import Decoder
from trainer.dalle2_decoder_trainer import DecoderTrainer
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from models.clip_model import OpenAIClipAdapter
from torchvision import transforms
from torchvision.datasets.coco import CocoCaptions
from tqdm import tqdm
import os
from configs.dalle2_configs import TrainDecoderConfig
from mydataset.dalle2_dataset import Dalle2Dataset, collate_func, collate_decoder_func
from datetime import timedelta
OUTPUT_DIR = './output'


def train_decoder(decoder_trainer, train_dataloader, device, config):
    total_loss = 0
    step = 0
    for batch in tqdm(train_dataloader):
        for unet_number in range(len(config.decoder.unets)):
            image, text = batch[0], batch[1]
            loss = decoder_trainer(image, unet_number=unet_number + 1)

            decoder_trainer.accelerator.backward(loss)
            decoder_trainer.update(unet_number=unet_number + 1)

            total_loss += loss.item()

        if step % 100 == 0 and decoder_trainer.accelerator.is_main_process:
            save_path = os.path.join(OUTPUT_DIR, 'decoder', 'model.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(decoder_trainer.accelerator.unwrap_model(decoder_trainer.decoder).state_dict(), save_path)
            print(f"average loss: {total_loss / (step + 1)}")

        step += 1

    print(step)


def main(config):
    # init
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True,
                                               static_graph=False)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60 * 60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])
    set_seed(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = accelerator.device
    print(device)
    print(accelerator.num_processes)

    # build model
    clip = OpenAIClipAdapter(config.decoder.clip.model).to(device)
    unet1 = Unet(**config.decoder.unets[0].dict(), cond_on_image_embeds=True).to(device)
    unet2 = Unet(**config.decoder.unets[1].dict(), lowres_cond=True).to(device)
    decoder_kwargs = config.decoder.dict()
    decoder_kwargs.pop('clip')
    decoder_kwargs.pop('unets')
    decoder = Decoder(unet=(unet1, unet2), clip=clip, **decoder_kwargs).to(device)

    decoder_trainer = DecoderTrainer(
        decoder,
        accelerator=accelerator,
        lr=3e-4,
        wd=1e-2,
        ema_beta=0.99,
        ema_update_after_step=1000,
        ema_update_every=10
    )

    # build dataloader
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    dataset = Dalle2Dataset(config.train.train_img_path, config.train.train_annot_path, transform)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=16, collate_fn=collate_decoder_func, drop_last=True)
    dataloader = accelerator.prepare(dataloader)

    for epoch in range(config.train.epochs):
        train_decoder(decoder_trainer, dataloader, device, config)


if __name__ == '__main__':
    config = TrainDecoderConfig.from_json_path('./configs/train_decoder_config.json')
    print(config)

    main(config)