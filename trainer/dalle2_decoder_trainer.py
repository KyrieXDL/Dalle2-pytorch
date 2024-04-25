import torch
import torch.nn as nn
from models.dalle2_decoder import Decoder
from dalle2_pytorch.train import groupby_prefix_and_trim, cast_tuple, get_optimizer, GradScaler, exists, autocast
from ema_pytorch import EMA
from functools import partial


class DecoderTrainer(nn.Module):
    def __init__(
        self,
        decoder,
        accelerator=None,
        use_ema = True,
        lr = 3e-4,
        wd = 1e-2,
        max_grad_norm = None,
        amp = False,
        **kwargs
    ):
        super().__init__()
        assert isinstance(decoder, Decoder)
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)

        self.accelerator = accelerator
        # self.decoder = decoder
        self.num_unets = len(decoder.unets)

        self.use_ema = use_ema

        if use_ema:
            has_lazy_linear = any([type(module) == nn.LazyLinear for module in decoder.modules()])
            assert not has_lazy_linear, 'you must set the text_embed_dim on your u-nets if you plan on doing automatic exponential moving average'

        self.ema_unets = nn.ModuleList([])

        self.amp = amp

        # be able to finely customize learning rate, weight decay
        # per unet

        lr, wd = map(partial(cast_tuple, length = self.num_unets), (lr, wd))

        optimizers = []
        for ind, (unet, unet_lr, unet_wd) in enumerate(zip(decoder.unets, lr, wd)):
            optimizer = get_optimizer(
                unet.parameters(),
                lr = unet_lr,
                wd = unet_wd,
            )
            optimizers.append(optimizer)
            # optimizer = self.accelerator.prepare([optimizer])[0]
            # setattr(self, f'optim{ind}', optimizer) # cannot use pytorch ModuleList for some reason with optimizers

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

            # scaler = GradScaler(enabled = amp)
            # setattr(self, f'scaler{ind}', scaler)

        # gradient clipping if needed
        self.max_grad_norm = max_grad_norm

        # accelerate prepare
        decoder, *optimizers = list(self.accelerator.prepare(decoder, *optimizers))

        self.decoder = decoder
        for opt_ind, optimizer in zip(range(len(optimizers)), optimizers):
            setattr(self, f'optim{opt_ind}', optimizer)

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    # def scale(self, loss, *, unet_number):
    #     assert 1 <= unet_number <= self.num_unets
    #     index = unet_number - 1
    #     scaler = getattr(self, f'scaler{index}')
    #     return scaler.scale(loss)

    def update(self, unet_number):
        assert 1 <= unet_number <= self.num_unets
        index = unet_number - 1
        unet = self.accelerator.unwrap_model(self.decoder).unets[index]

        optimizer = getattr(self, f'optim{index}')
        # scaler = getattr(self, f'scaler{index}')

        # if exists(self.max_grad_norm):
        #     scaler.unscale_(optimizer)
        #     nn.utils.clip_grad_norm_(unet.parameters(), self.max_grad_norm)
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)  # Automatically unscales gradients

        optimizer.step()
        optimizer.zero_grad()

        if self.use_ema:
            ema_unet = self.ema_unets[index]
            ema_unet.update()

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        if self.use_ema:
            trainable_unets = self.decoder.unets
            self.decoder.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = self.decoder.sample(*args, **kwargs)

        if self.use_ema:
            self.decoder.unets = trainable_unets             # restore original training unets
        return output

    def forward(
        self,
        x,
        *,
        unet_number,
        divisor = 1,
        **kwargs
    ):
        # with autocast(enabled = self.amp):
        #     loss = self.decoder(x, unet_number = unet_number, **kwargs)
        # return self.scale(loss / divisor, unet_number = unet_number)

        with self.accelerator.autocast():
            loss = self.decoder(x, unet_number=unet_number, **kwargs)

        return loss
        # return self.scale(loss / divisor, unet_number=unet_number)
