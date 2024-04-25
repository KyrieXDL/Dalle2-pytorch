import torch
from torchvision import transforms as T
from pathlib import Path
import os
from tqdm import tqdm
from models.unet import Unet
from models.dalle2_decoder import Decoder
from configs.dalle2_configs import TrainDecoderConfig, TrainDiffusionPriorConfig
from models.clip_model import OpenAIClipAdapter
from models.dalle2 import DALLE2
from models.dalle2_prior import DiffusionPriorNetwork, DiffusionPrior
from torch.utils.data import DataLoader
from mydataset.dalle2_dataset import Dalle2Dataset, collate_decoder_func
import argparse


def main(args):
    input_image_size = args.input_image_size
    test_img_path = args.test_img_path
    test_annot_path = args.test_annot_path
    device = "cuda"
    test_img_save_path = "./result"

    if not os.path.exists(test_img_save_path):
        os.makedirs(test_img_save_path)

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(input_image_size),
        T.CenterCrop(input_image_size),
        T.ToTensor()
    ])

    dataset = Dalle2Dataset(test_img_path, test_annot_path, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, collate_fn=collate_decoder_func,
                            drop_last=True)

    ### prior
    config = TrainDiffusionPriorConfig.from_json_path(args.prior_config)
    clip = OpenAIClipAdapter(config.prior.clip.model).to(device)
    prior_network = DiffusionPriorNetwork(**config.prior.net.dict()).to(device)
    prior_kwargs = config.prior.dict()
    prior_kwargs.pop('clip')
    prior_kwargs.pop('net')
    diffusion_prior = DiffusionPrior(net=prior_network, clip=clip, **prior_kwargs).to(device)
    diffusion_prior.load_state_dict(torch.load(args.prior_ckpt))

    ### decoder
    config = TrainDecoderConfig.from_json_path(args.decoder_config)
    unet1 = Unet(**config.decoder.unets[0].dict(), cond_on_image_embeds=True).to(device)
    unet2 = Unet(**config.decoder.unets[1].dict(), lowres_cond=True).to(device)
    decoder_kwargs = config.decoder.dict()
    decoder_kwargs.pop('clip')
    decoder_kwargs.pop('unets')
    decoder = Decoder(unet=(unet1, unet2), clip=clip, **decoder_kwargs).to(device)
    decoder.load_state_dict(torch.load(args.decoder_config))

    dalle2 = DALLE2(
        prior=diffusion_prior,
        decoder=decoder
    ).to(device)

    idx = 0

    all_texts = []
    for data in tqdm(dataloader):
        image, target, text = data
        target = target.to(device)

        all_texts += text

        orig_image = image[0]
        orig_image = T.ToPILImage()(orig_image)
        test_save_path = test_img_save_path + "/" + str(idx) + "_orig.jpg"
        orig_image.save(Path(test_save_path))

        test_img_tensors = dalle2(
            target,
            cond_scale=3.,  # classifier free guidance strength (> 1 would strengthen the condition)
        )
        # print(test_img_tensors.shape)
        # print(torch.max(test_img_tensors), torch.min(test_img_tensors))

        test_img = T.ToPILImage()(test_img_tensors)
        test_save_path = test_img_save_path + "/" + str(idx) + ".jpg"
        test_img.save(Path(test_save_path))

        idx += 1
        if idx > 50:
            break

    with open('./result/texts.txt', 'w') as fw:
        fw.write('\n'.join(all_texts))


if __name__ == '__main__':
    argument = argparse.ArgumentParser()
    argument.add_argument('--test_img_path', type=str, default='./data/coco/val2014')
    argument.add_argument('--test_annot_path', type=str, default='./data/coco/annotations_trainval2014/annotations/captions_val2014.json')
    argument.add_argument('--input_image_size', type=int, default=256)
    argument.add_argument('--prior_config', type=str, default='./configs/train_prior_config.json')
    argument.add_argument('--decoder_config', type=str, default='./configs/train_decoder_config.json')
    argument.add_argument('--prior_ckpt', type=str, default='./output/prior/model.pt')
    argument.add_argument('--decoder_ckpt', type=str, default='./output/decoder/model.pt')

    args = argument.parse_args()

    main(args)