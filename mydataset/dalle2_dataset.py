import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import os
from dalle2_pytorch.tokenizer import tokenizer
from torchvision import transforms


def normalize_img(img):
    return img * 2 - 1


def unnormalize_img(normed_img):
    return (normed_img + 1) * 0.5


class Dalle2Dataset(Dataset):
    def __init__(self, image_dir, annFile, transform, norm=False):
        super().__init__()
        self.root = image_dir
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.norm = norm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        cur_id = self.ids[index]
        path = self.coco.loadImgs(cur_id)[0]["file_name"]
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        image = self.transform(image)

        # if self.norm:
        # image = normalize_img(image) # convert (0,1) to (-1, 1)

        texts = [ann['caption'] for ann in self.coco.loadAnns(self.coco.getAnnIds(cur_id))]
        texts_ids = tokenizer.tokenize(texts)

        image = image.unsqueeze(0).repeat(len(texts), 1, 1, 1)

        return {'image': image, 'text': texts_ids, 'raw_text': texts}


def collate_func(batch):
    image = [item['image'] for item in batch]
    text = [item['text'] for item in batch]

    image = torch.cat(image, dim=0)
    text = torch.cat(text, dim=0)

    return image, text


def collate_decoder_func(batch):
    image = [item['image'][0] for item in batch]
    text = [item['text'][0] for item in batch]
    raw_text = [item['raw_text'][0] for item in batch]

    image = torch.stack(image, dim=0)
    text = torch.stack(text, dim=0)

    return image, text, raw_text


if __name__ == '__main__':
    annFile = '../data/coco/annotations_trainval2014/annotations/captions_train2014.json'
    # coco = COCO(annFile)
    # ids = list(sorted(coco.imgs.keys()))
    # print(len(ids))
    # print(ids[:10])
    #
    # texts = [ann['caption'] for ann in coco.loadAnns(coco.getAnnIds(ids[0]))]
    # print(texts)
    #
    # path = coco.loadImgs(ids[0])
    # print(path)
    # print(len(path))

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    dataset = Dalle2Dataset('../data/coco/train2014', annFile, transform)

    # image, text = dataset[0]
    # print(image.shape, text.shape)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_decoder_func)
    for image, text in dataloader:
        print(image.shape, text.shape)
        break