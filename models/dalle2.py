import torch
import torch.nn as nn
import torchvision.transforms as T
from dalle2_pytorch.tokenizer import tokenizer


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class DALLE2(nn.Module):
    def __init__(
        self,
        *,
        prior,
        decoder,
        prior_num_samples = 2
    ):
        super().__init__()
        self.prior = prior
        self.decoder = decoder

        self.prior_num_samples = prior_num_samples
        self.decoder_need_text_cond = self.decoder.condition_on_text_encodings

        self.to_pil = T.ToPILImage()

    @torch.inference_mode()
    @eval_decorator
    def forward(
        self,
        text,
        cond_scale = 1.,
        return_pil_images = False
    ):
        device = next(self.parameters()).device
        one_text = isinstance(text, str) or (not is_list_str(text) and text.shape[0] == 1)

        if isinstance(text, str) or is_list_str(text):
            text = [text] if not isinstance(text, (list, tuple)) else text
            text = tokenizer.tokenize(text).to(device)

        image_embed = self.prior.sample(text, num_samples_per_batch = self.prior_num_samples)
        # image_embed = text
        text_cond = text if self.decoder_need_text_cond else None
        images = self.decoder.sample(image_embed=image_embed, text = text_cond, cond_scale = cond_scale)

        if return_pil_images:
            images = list(map(self.to_pil, images.unbind(dim = 0)))

        if one_text:
            return images[0]

        return images
