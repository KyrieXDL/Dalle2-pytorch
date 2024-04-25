import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom
from functools import partial
from functools import partial, wraps
from torch.utils.checkpoint import checkpoint
from models.dalle2_utils import default, exists, CrossAttention, Residual, LinearAttention, SinusoidalPosEmb, Attention, prob_mask_like, cast_tuple, resize_image_to


def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner


def identity(t, *args, **kwargs):
    return t


def make_checkpointable(fn, **kwargs):
    if isinstance(fn, nn.ModuleList):
        return [maybe(make_checkpointable)(el, **kwargs) for el in fn]

    condition = kwargs.pop('condition', None)

    if exists(condition) and not condition(fn):
        return fn

    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner


def NearestUpsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )


class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)


class RearrangeToSequence(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = rearrange(x, 'b c ... -> b ... c')
        x, ps = pack([x], 'b * c')

        x = self.fn(x)

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b ... c -> b c ...')
        return x


class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    ):
        super().__init__()
        assert len(dim_ins) == len(dim_outs)
        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps = None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim = 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        flattened_weights = rearrange(weight, 'o ... -> o (...)')

        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')

        var = torch.var(flattened_weights, dim = -1, unbiased = False)
        var = rearrange(var, 'o -> o 1 1 1')

        weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        weight_standardization = False
    ):
        super().__init__()
        conv_klass = nn.Conv2d if not weight_standardization else WeightStandardizedConv2d

        self.project = conv_klass(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.project(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8,
        weight_standardization = False,
        cosine_sim_cross_attn = False
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            self.cross_attn = CrossAttention(
                dim = dim_out,
                context_dim = cond_dim,
                cosine_sim = cosine_sim_cross_attn
            )

        self.block1 = Block(dim, dim_out, groups = groups, weight_standardization = weight_standardization)
        self.block2 = Block(dim_out, dim_out, groups = groups, weight_standardization = weight_standardization)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, cond = None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        if exists(self.cross_attn):
            assert exists(cond)

            h = rearrange(h, 'b c ... -> b ... c')
            h, ps = pack([h], 'b * c')

            h = self.cross_attn(h, context = cond) + h

            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b ... c -> b c ...')

        h = self.block2(h)
        return h + self.res_conv(x)


def Downsample(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_embed_dim = None,
        text_embed_dim = None,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        channels_out = None,
        self_attn = False,
        attn_dim_head = 32,
        attn_heads = 16,
        lowres_cond = False,             # for cascading diffusion - https://cascaded-diffusion.github.io/
        lowres_noise_cond = False,       # for conditioning on low resolution noising, based on Imagen
        self_cond = False,               # set this to True to use the self-conditioning technique from - https://arxiv.org/abs/2208.04202
        sparse_attn = False,
        cosine_sim_cross_attn = False,
        cosine_sim_self_attn = False,
        attend_at_middle = True,         # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        cond_on_text_encodings = False,
        max_text_len = 256,
        cond_on_image_embeds = False,
        add_image_embeds_to_time = True, # alerted by @mhh0318 to a phrase in the paper - "Specifically, we modify the architecture described in Nichol et al. (2021) by projecting and adding CLIP embeddings to the existing timestep embedding"
        init_dim = None,
        init_conv_kernel_size = 7,
        resnet_groups = 8,
        resnet_weight_standardization = False,
        num_resnet_blocks = 2,
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        memory_efficient = False,
        scale_skip_connection = False,
        pixel_shuffle_upsample = True,
        final_conv_kernel_size = 1,
        combine_upsample_fmaps = False, # whether to combine the outputs of all upsample blocks, as in unet squared paper
        checkpoint_during_training = False,
        **kwargs
    ):
        super().__init__()
        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        del self._locals['self']
        del self._locals['__class__']

        # for eventual cascading diffusion
        self.lowres_cond = lowres_cond

        # whether to do self conditioning
        self.self_cond = self_cond

        # determine dimensions
        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # initial number of channels depends on
        # (1) low resolution conditioning from cascading ddpm paper, conditioned on previous unet output in the cascade
        # (2) self conditioning (bit diffusion paper)
        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))
        init_dim = default(init_dim, dim)

        print(f'init: channels: {init_channels}')

        self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1) if init_cross_embed else nn.Conv2d(init_channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        num_stages = len(in_out)

        # time, image embeddings, and optional text encoding
        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4

        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.GELU()
        )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        self.image_to_tokens = nn.Sequential(
            nn.Linear(image_embed_dim, cond_dim * num_image_tokens),
            Rearrange('b (n d) -> b n d', n = num_image_tokens)
        ) if cond_on_image_embeds and image_embed_dim != cond_dim else nn.Identity()

        self.to_image_hiddens = nn.Sequential(
            nn.Linear(image_embed_dim, time_cond_dim),
            nn.GELU()
        ) if cond_on_image_embeds and add_image_embeds_to_time else None

        self.norm_cond = nn.LayerNorm(cond_dim)
        self.norm_mid_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)
        self.text_to_cond = None
        self.text_embed_dim = None

        if cond_on_text_encodings:
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text_encodings is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)
            self.text_embed_dim = text_embed_dim

        # low resolution noise conditiong, based on Imagen's upsampler training technique
        self.lowres_noise_cond = lowres_noise_cond
        self.to_lowres_noise_cond = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.GELU(),
            nn.Linear(time_cond_dim, time_cond_dim)
        ) if lowres_noise_cond else None

        # finer control over whether to condition on image embeddings and text encodings
        # so one can have the latter unets in the cascading DDPMs only focus on super-resoluting
        self.cond_on_text_encodings = cond_on_text_encodings
        self.cond_on_image_embeds = cond_on_image_embeds

        # for classifier free guidance
        self.null_image_embed = nn.Parameter(torch.randn(1, num_image_tokens, cond_dim))
        self.null_image_hiddens = nn.Parameter(torch.randn(1, time_cond_dim))

        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))

        # whether to scale skip connection, adopted in Imagen
        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # attention related params
        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head, cosine_sim = cosine_sim_self_attn)

        self_attn = cast_tuple(self_attn, num_stages)

        create_self_attn = lambda dim: RearrangeToSequence(Residual(Attention(dim, **attn_kwargs)))

        # resnet block klass
        resnet_groups = cast_tuple(resnet_groups, num_stages)
        top_level_resnet_group = first(resnet_groups)

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_stages)

        # downsample klass
        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # upsample klass
        upsample_klass = NearestUpsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # prepare resnet klass
        resnet_block = partial(ResnetBlock, cosine_sim_cross_attn = cosine_sim_cross_attn, weight_standardization = resnet_weight_standardization)

        # give memory efficient unet an initial resnet block
        self.init_resnet_block = resnet_block(init_dim, init_dim, time_cond_dim = time_cond_dim, groups = top_level_resnet_group) if memory_efficient else None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        skip_connect_dims = []          # keeping track of skip connection dimensions
        upsample_combiner_dims = []     # keeping track of dimensions for final upsample feature map combiner

        for ind, ((dim_in, dim_out), groups, layer_num_resnet_blocks, layer_self_attn) in enumerate(zip(in_out, resnet_groups, num_resnet_blocks, self_attn)):
            is_first = ind == 0
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if not is_first else None

            dim_layer = dim_out if memory_efficient else dim_in
            skip_connect_dims.append(dim_layer)

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_layer)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_layer, **attn_kwargs))

            self.downs.append(nn.ModuleList([
                downsample_klass(dim_in, dim_out = dim_out) if memory_efficient else None,
                resnet_block(dim_layer, dim_layer, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([resnet_block(dim_layer, dim_layer, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups) for _ in range(layer_num_resnet_blocks)]),
                attention,
                downsample_klass(dim_layer, dim_out = dim_out) if not is_last and not memory_efficient else nn.Conv2d(dim_layer, dim_out, 1)
            ]))

        mid_dim = dims[-1]

        self.mid_block1 = resnet_block(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])
        self.mid_attn = create_self_attn(mid_dim)
        self.mid_block2 = resnet_block(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])

        for ind, ((dim_in, dim_out), groups, layer_num_resnet_blocks, layer_self_attn) in enumerate(zip(reversed(in_out), reversed(resnet_groups), reversed(num_resnet_blocks), reversed(self_attn))):
            is_last = ind >= (len(in_out) - 1)
            layer_cond_dim = cond_dim if not is_last else None

            skip_connect_dim = skip_connect_dims.pop()

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_out)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_out, **attn_kwargs))

            upsample_combiner_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_block(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([resnet_block(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups)  for _ in range(layer_num_resnet_blocks)]),
                attention,
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else nn.Identity()
            ]))

        # whether to combine outputs from all upsample blocks for final resnet block
        self.upsample_combiner = UpsampleCombiner(
            dim = dim,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_combiner_dims,
            dim_outs = (dim,) * len(upsample_combiner_dims)
        )

        # a final resnet block
        self.final_resnet_block = resnet_block(self.upsample_combiner.dim_out + dim, dim, time_cond_dim = time_cond_dim, groups = top_level_resnet_group)
        out_dim_in = dim + (channels if lowres_cond else 0)
        self.to_out = nn.Conv2d(out_dim_in, self.channels_out, kernel_size = final_conv_kernel_size, padding = final_conv_kernel_size // 2)
        zero_init_(self.to_out) # since both OpenAI and @crowsonkb are doing it

        # whether to checkpoint during training
        self.checkpoint_during_training = checkpoint_during_training

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        lowres_noise_cond,
        channels,
        channels_out,
        cond_on_image_embeds,
        cond_on_text_encodings,
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_image_embeds == self.cond_on_image_embeds and \
            cond_on_text_encodings == self.cond_on_text_encodings and \
            lowres_noise_cond == self.lowres_noise_cond and \
            channels_out == self.channels_out:
            return self
        print(cond_on_image_embeds, self.cond_on_image_embeds, cond_on_text_encodings, self.cond_on_text_encodings, channels_out, self.channels_out)
        print(f'Reinit: channels: {channels}')
        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            channels = channels,
            channels_out = channels_out,
            cond_on_image_embeds = cond_on_image_embeds,
            cond_on_text_encodings = cond_on_text_encodings,
            lowres_noise_cond = lowres_noise_cond
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        image_embed,
        lowres_cond_img = None,
        lowres_noise_level = None,
        text_encodings = None,
        image_cond_drop_prob = 0.,
        text_cond_drop_prob = 0.,
        blur_sigma = None,
        blur_kernel_size = None,
        disable_checkpoint = False,
        self_cond = None
    ):
        batch_size, device = x.shape[0], x.device

        # add low resolution conditioning, if present
        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'

        # concat self conditioning, if needed
        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim = 1)

        # concat low resolution conditioning
        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # initial convolution
        # print('Before init conv: ', x.shape)
        x = self.init_conv(x)
        r = x.clone() # final residual
        # print('After init conv', x.shape)

        # time conditioning
        time = time.type_as(x)
        time_hiddens = self.to_time_hiddens(time)
        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # low res noise conditioning (similar to time above)
        if exists(lowres_noise_level):
            assert exists(self.to_lowres_noise_cond), 'lowres_noise_cond must be set to True on instantiation of the unet in order to conditiong on lowres noise'
            lowres_noise_level = lowres_noise_level.type_as(x)
            t = t + self.to_lowres_noise_cond(lowres_noise_level)

        # conditional dropout
        image_keep_mask = prob_mask_like((batch_size,), 1 - image_cond_drop_prob, device = device)
        text_keep_mask = prob_mask_like((batch_size,), 1 - text_cond_drop_prob, device = device)
        text_keep_mask = rearrange(text_keep_mask, 'b -> b 1 1')

        # image embedding to be summed to time embedding
        # discovered by @mhh0318 in the paper
        if exists(image_embed) and exists(self.to_image_hiddens):
            image_hiddens = self.to_image_hiddens(image_embed)
            image_keep_mask_hidden = rearrange(image_keep_mask, 'b -> b 1')
            null_image_hiddens = self.null_image_hiddens.to(image_hiddens.dtype)

            image_hiddens = torch.where(
                image_keep_mask_hidden,
                image_hiddens,
                null_image_hiddens
            )

            t = t + image_hiddens

        # mask out image embedding depending on condition dropout
        # for classifier free guidance
        image_tokens = None

        if self.cond_on_image_embeds:
            image_keep_mask_embed = rearrange(image_keep_mask, 'b -> b 1 1')
            image_tokens = self.image_to_tokens(image_embed)
            null_image_embed = self.null_image_embed.to(image_tokens.dtype) # for some reason pytorch AMP not working

            image_tokens = torch.where(
                image_keep_mask_embed,
                image_tokens,
                null_image_embed
            )

        # take care of text encodings (optional)
        text_tokens = None

        if exists(text_encodings) and self.cond_on_text_encodings:
            assert text_encodings.shape[0] == batch_size, f'the text encodings being passed into the unet does not have the proper batch size - text encoding shape {text_encodings.shape} - required batch size is {batch_size}'
            assert self.text_embed_dim == text_encodings.shape[-1], f'the text encodings you are passing in have a dimension of {text_encodings.shape[-1]}, but the unet was created with text_embed_dim of {self.text_embed_dim}.'

            text_mask = torch.any(text_encodings != 0., dim = -1)

            text_tokens = self.text_to_cond(text_encodings)

            text_tokens = text_tokens[:, :self.max_text_len]
            text_mask = text_mask[:, :self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
                text_mask = F.pad(text_mask, (0, remainder), value = False)

            text_mask = rearrange(text_mask, 'b n -> b n 1')

            assert text_mask.shape[0] == text_keep_mask.shape[0], f'text_mask has shape of {text_mask.shape} while text_keep_mask has shape {text_keep_mask.shape}. text encoding is of shape {text_encodings.shape}'
            text_keep_mask = text_mask & text_keep_mask

            null_text_embed = self.null_text_embed.to(text_tokens.dtype) # for some reason pytorch AMP not working

            text_tokens = torch.where(
                text_keep_mask,
                text_tokens,
                null_text_embed
            )

        # main conditioning tokens (c)
        c = time_tokens
        if exists(image_tokens):
            c = torch.cat((c, image_tokens), dim = -2)

        # text and image conditioning tokens (mid_c)
        # to save on compute, only do cross attention based conditioning on the inner most layers of the Unet
        mid_c = c if not exists(text_tokens) else torch.cat((c, text_tokens), dim = -2)

        # normalize conditioning tokens
        c = self.norm_cond(c)
        mid_c = self.norm_mid_cond(mid_c)

        # gradient checkpointing
        can_checkpoint = self.training and self.checkpoint_during_training and not disable_checkpoint
        apply_checkpoint_fn = make_checkpointable if can_checkpoint else identity

        # make checkpointable modules
        init_resnet_block, mid_block1, mid_attn, mid_block2, final_resnet_block = [maybe(apply_checkpoint_fn)(module) for module in (self.init_resnet_block, self.mid_block1, self.mid_attn, self.mid_block2, self.final_resnet_block)]
        can_checkpoint_cond = lambda m: isinstance(m, ResnetBlock)
        downs, ups = [maybe(apply_checkpoint_fn)(m, condition = can_checkpoint_cond) for m in (self.downs, self.ups)]

        # initial resnet block
        if exists(init_resnet_block):
            x = init_resnet_block(x, t)

        # go through the layers of the unet, down and up
        down_hiddens = []
        up_hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn, post_downsample in downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t, c)
                down_hiddens.append(x.contiguous())

            x = attn(x)
            down_hiddens.append(x.contiguous())

            if exists(post_downsample):
                x = post_downsample(x)

        x = mid_block1(x, t, mid_c)
        if exists(mid_attn):
            x = mid_attn(x)
        x = mid_block2(x, t, mid_c)

        connect_skip = lambda fmap: torch.cat((fmap, down_hiddens.pop() * self.skip_connect_scale), dim = 1)

        for init_block, resnet_blocks, attn, upsample in ups:
            x = connect_skip(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = connect_skip(x)
                x = resnet_block(x, t, c)

            x = attn(x)

            up_hiddens.append(x.contiguous())
            x = upsample(x)

        x = self.upsample_combiner(x, up_hiddens)
        x = torch.cat((x, r), dim = 1)
        x = final_resnet_block(x, t)

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        return self.to_out(x)