import torch
import torch.nn as nn
from models.dalle2_utils import (
    exists,
    default,
    cast_tuple,
    pad_tuple_to_length,
    NullVQGanVAE,
    VQGanVAE,
    LowresConditioner,
    eval_decorator,
    null_context,
    maybe,
    normalize_neg_one_to_one,
    identity,
    unnormalize_zero_to_one,
    freeze_model_and_make_eval_,
    NoiseScheduler,
    module_device,
    UnetOutput,
    rearrange,
    extract,
    reduce,
    meanflat,
    discretized_gaussian_log_likelihood,
    normal_kl,
    NAT,
    resize_image_to)
import random
from models.unet import Unet
from contextlib import contextmanager
from collections import namedtuple
from tqdm import tqdm
import kornia.augmentation as K


class Decoder(nn.Module):
    def __init__(
            self,
            unet,
            *,
            clip=None,
            image_size=None,
            channels=3,
            vae=tuple(),
            timesteps=1000,
            sample_timesteps=None,
            image_cond_drop_prob=0.1,
            text_cond_drop_prob=0.5,
            loss_type='l2',
            beta_schedule=None,
            predict_x_start=False,
            predict_v=False,
            predict_x_start_for_latent_diffusion=False,
            image_sizes=None,  # for cascading ddpm, image size at each stage
            random_crop_sizes=None,
            # whether to random crop the image at that stage in the cascade (super resoluting convolutions at the end may be able to generalize on smaller crops)
            use_noise_for_lowres_cond=False,  # whether to use Imagen-like noising for low resolution conditioning
            use_blur_for_lowres_cond=True,
            # whether to use the blur conditioning used in the original cascading ddpm paper, as well as DALL-E2
            lowres_downsample_first=True,
            # cascading ddpm - resizes to lower resolution, then to next conditional resolution + blur
            blur_prob=0.5,  # cascading ddpm - when training, the gaussian blur is only applied 50% of the time
            blur_sigma=0.6,  # cascading ddpm - blur sigma
            blur_kernel_size=3,  # cascading ddpm - blur kernel size
            lowres_noise_sample_level=0.2,
            # in imagen paper, they use a 0.2 noise level at sample time for low resolution conditioning
            clip_denoised=True,
            clip_x_start=True,
            clip_adapter_overrides=dict(),
            learned_variance=True,
            learned_variance_constrain_frac=False,
            vb_loss_weight=0.001,
            unconditional=False,  # set to True for generating images without conditioning
            auto_normalize_img=True,
            # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
            use_dynamic_thres=False,  # from the Imagen paper
            dynamic_thres_percentile=0.95,
            p2_loss_weight_gamma=0.,
            # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k=1,
            ddim_sampling_eta=0.  # can be set to 0. for deterministic sampling afaict
    ):
        super().__init__()

        # clip
        freeze_model_and_make_eval_(clip)
        self.clip = clip

        # determine image size, with image_size and image_sizes taking precedence
        if exists(image_size) or exists(image_sizes):
            assert exists(image_size) ^ exists(image_sizes), 'only one of image_size or image_sizes must be given'
            image_size = default(image_size, lambda: image_sizes[-1])
        elif exists(clip):
            image_size = clip.image_size
        else:
            raise ValueError('either image_size, image_sizes, or clip must be given to decoder')

        # channels
        self.channels = channels

        # normalize and unnormalize image functions
        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity

        # verify conditioning method
        unets = cast_tuple(unet)
        num_unets = len(unets)
        self.num_unets = num_unets

        self.unconditional = unconditional

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet
        vaes = pad_tuple_to_length(cast_tuple(vae), len(unets), fillvalue=NullVQGanVAE(channels=self.channels))

        # whether to use learned variance, defaults to True for the first unet in the cascade, as in paper
        learned_variance = pad_tuple_to_length(cast_tuple(learned_variance), len(unets), fillvalue=False)
        self.learned_variance = learned_variance
        self.learned_variance_constrain_frac = learned_variance_constrain_frac  # whether to constrain the output of the network (the interpolation fraction) from 0 to 1
        self.vb_loss_weight = vb_loss_weight

        # default and validate conditioning parameters
        use_noise_for_lowres_cond = cast_tuple(use_noise_for_lowres_cond, num_unets - 1, validate=False)
        use_blur_for_lowres_cond = cast_tuple(use_blur_for_lowres_cond, num_unets - 1, validate=False)

        if len(use_noise_for_lowres_cond) < num_unets:
            use_noise_for_lowres_cond = (False, *use_noise_for_lowres_cond)

        if len(use_blur_for_lowres_cond) < num_unets:
            use_blur_for_lowres_cond = (False, *use_blur_for_lowres_cond)

        assert not use_noise_for_lowres_cond[0], 'first unet will never need low res noise conditioning'
        assert not use_blur_for_lowres_cond[0], 'first unet will never need low res blur conditioning'

        assert num_unets == 1 or all((use_noise or use_blur) for use_noise, use_blur in
                                     zip(use_noise_for_lowres_cond[1:], use_blur_for_lowres_cond[1:]))

        # construct unets and vaes
        self.unets = nn.ModuleList([])
        self.vaes = nn.ModuleList([])

        for ind, (one_unet, one_vae, one_unet_learned_var, lowres_noise_cond) in enumerate(
                zip(unets, vaes, learned_variance, use_noise_for_lowres_cond)):
            assert isinstance(one_unet, Unet)
            assert isinstance(one_vae, (VQGanVAE, NullVQGanVAE))

            is_first = ind == 0
            latent_dim = one_vae.encoded_dim if exists(one_vae) else None

            unet_channels = default(latent_dim, self.channels)
            unet_channels_out = unet_channels * (1 if not one_unet_learned_var else 2)

            one_unet = one_unet.cast_model_parameters(
                lowres_cond=not is_first,
                lowres_noise_cond=lowres_noise_cond,
                cond_on_image_embeds=not unconditional and is_first,
                cond_on_text_encodings=not unconditional and one_unet.cond_on_text_encodings,
                channels=unet_channels,
                channels_out=unet_channels_out
            )

            self.unets.append(one_unet)
            self.vaes.append(one_vae.copy_for_eval())

        # sampling timesteps, defaults to non-ddim with full timesteps sampling
        self.sample_timesteps = cast_tuple(sample_timesteps, num_unets)
        self.ddim_sampling_eta = ddim_sampling_eta

        # create noise schedulers per unet
        if not exists(beta_schedule):
            beta_schedule = ('cosine', *(('cosine',) * max(num_unets - 2, 0)), *(('linear',) * int(num_unets > 1)))

        beta_schedule = cast_tuple(beta_schedule, num_unets)
        p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        self.noise_schedulers = nn.ModuleList([])

        for ind, (unet_beta_schedule, unet_p2_loss_weight_gamma, sample_timesteps) in enumerate(
                zip(beta_schedule, p2_loss_weight_gamma, self.sample_timesteps)):
            assert not exists(
                sample_timesteps) or sample_timesteps <= timesteps, f'sampling timesteps {sample_timesteps} must be less than or equal to the number of training timesteps {timesteps} for unet {ind + 1}'

            noise_scheduler = NoiseScheduler(
                beta_schedule=unet_beta_schedule,
                timesteps=timesteps,
                loss_type=loss_type,
                p2_loss_weight_gamma=unet_p2_loss_weight_gamma,
                p2_loss_weight_k=p2_loss_weight_k
            )

            self.noise_schedulers.append(noise_scheduler)

        # unet image sizes
        image_sizes = default(image_sizes, (image_size,))
        image_sizes = tuple(sorted(set(image_sizes)))

        assert self.num_unets == len(
            image_sizes), f'you did not supply the correct number of u-nets ({self.num_unets}) for resolutions {image_sizes}'
        self.image_sizes = image_sizes
        self.sample_channels = cast_tuple(self.channels, len(image_sizes))

        # random crop sizes (for super-resoluting unets at the end of cascade?)
        self.random_crop_sizes = cast_tuple(random_crop_sizes, len(image_sizes))
        assert not exists(self.random_crop_sizes[0]), 'you would not need to randomly crop the image for the base unet'

        # predict x0 config
        self.predict_x_start = cast_tuple(predict_x_start,
                                          len(unets)) if not predict_x_start_for_latent_diffusion else tuple(
            map(lambda t: isinstance(t, VQGanVAE), self.vaes))

        # predict v
        self.predict_v = cast_tuple(predict_v, len(unets))

        # input image range
        self.input_image_range = (-1. if not auto_normalize_img else 0., 1.)

        # cascading ddpm related stuff
        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (
                num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_conds = nn.ModuleList([])

        for unet_index, use_noise, use_blur in zip(range(num_unets), use_noise_for_lowres_cond,
                                                   use_blur_for_lowres_cond):
            if unet_index == 0:
                self.lowres_conds.append(None)
                continue

            lowres_cond = LowresConditioner(
                downsample_first=lowres_downsample_first,
                use_blur=use_blur,
                use_noise=use_noise,
                blur_prob=blur_prob,
                blur_sigma=blur_sigma,
                blur_kernel_size=blur_kernel_size,
                input_image_range=self.input_image_range,
                normalize_img_fn=self.normalize_img,
                unnormalize_img_fn=self.unnormalize_img
            )

            self.lowres_conds.append(lowres_cond)

        self.lowres_noise_sample_level = lowres_noise_sample_level

        # classifier free guidance

        self.image_cond_drop_prob = image_cond_drop_prob
        self.text_cond_drop_prob = text_cond_drop_prob
        self.can_classifier_guidance = image_cond_drop_prob > 0. or text_cond_drop_prob > 0.

        # whether to clip when sampling

        self.clip_denoised = clip_denoised
        self.clip_x_start = clip_x_start

        # dynamic thresholding settings, if clipping denoised during sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        # device tracker

        self.register_buffer('_dummy', torch.Tensor([True]), persistent=False)

    @property
    def device(self):
        return self._dummy.device

    @property
    def condition_on_text_encodings(self):
        return any([unet.cond_on_text_encodings for unet in self.unets if isinstance(unet, Unet)])

    def get_unet(self, unet_number):
        assert 0 < unet_number <= self.num_unets
        index = unet_number - 1
        return self.unets[index]

    def parse_unet_output(self, learned_variance, output):
        var_interp_frac_unnormalized = None

        if learned_variance:
            output, var_interp_frac_unnormalized = output.chunk(2, dim=1)

        return UnetOutput(output, var_interp_frac_unnormalized)

    @contextmanager
    def one_unet_in_gpu(self, unet_number=None, unet=None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.get_unet(unet_number)

        # devices

        cuda, cpu = torch.device('cuda'), torch.device('cpu')

        self.cuda()

        devices = [module_device(unet) for unet in self.unets]

        self.unets.to(cpu)
        unet.to(cuda)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    def dynamic_threshold(self, x):
        """ proposed in https://arxiv.org/abs/2205.11487 as an improved clamping in the setting of classifier free guidance """

        # s is the threshold amount
        # static thresholding would just be s = 1
        s = 1.
        if self.use_dynamic_thres:
            s = torch.quantile(
                rearrange(x, 'b ... -> b (...)').abs(),
                self.dynamic_thres_percentile,
                dim=-1
            )

            s.clamp_(min=1.)
            s = s.view(-1, *((1,) * (x.ndim - 1)))

        # clip by threshold, depending on whether static or dynamic
        x = x.clamp(-s, s) / s
        return x

    def p_mean_variance(self, unet, x, t, image_embed, noise_scheduler, text_encodings=None, lowres_cond_img=None,
                        self_cond=None, clip_denoised=True, predict_x_start=False, predict_v=False,
                        learned_variance=False, cond_scale=1., model_output=None, lowres_noise_level=None):
        assert not (
                cond_scale != 1. and not self.can_classifier_guidance), 'the decoder was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        model_output = default(model_output, lambda: unet.forward_with_cond_scale(x, t, image_embed=image_embed,
                                                                                  text_encodings=text_encodings,
                                                                                  cond_scale=cond_scale,
                                                                                  lowres_cond_img=lowres_cond_img,
                                                                                  self_cond=self_cond,
                                                                                  lowres_noise_level=lowres_noise_level))

        pred, var_interp_frac_unnormalized = self.parse_unet_output(learned_variance, model_output)

        if predict_v:
            x_start = noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif predict_x_start:
            x_start = pred
        else:
            x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if clip_denoised:
            x_start = self.dynamic_threshold(x_start)

        model_mean, posterior_variance, posterior_log_variance = noise_scheduler.q_posterior(x_start=x_start, x_t=x,
                                                                                             t=t)

        if learned_variance:
            # if learned variance, posterio variance and posterior log variance are predicted by the network
            # by an interpolation of the max and min log beta values
            # eq 15 - https://arxiv.org/abs/2102.09672
            min_log = extract(noise_scheduler.posterior_log_variance_clipped, t, x.shape)
            max_log = extract(torch.log(noise_scheduler.betas), t, x.shape)
            var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

            if self.learned_variance_constrain_frac:
                var_interp_frac = var_interp_frac.sigmoid()

            posterior_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
            posterior_variance = posterior_log_variance.exp()

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, unet, x, t, image_embed, noise_scheduler, text_encodings=None, cond_scale=1.,
                 lowres_cond_img=None, self_cond=None, predict_x_start=False, predict_v=False, learned_variance=False,
                 clip_denoised=True, lowres_noise_level=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(unet, x=x, t=t, image_embed=image_embed,
                                                                          text_encodings=text_encodings,
                                                                          cond_scale=cond_scale,
                                                                          lowres_cond_img=lowres_cond_img,
                                                                          self_cond=self_cond,
                                                                          clip_denoised=clip_denoised,
                                                                          predict_x_start=predict_x_start,
                                                                          predict_v=predict_v,
                                                                          noise_scheduler=noise_scheduler,
                                                                          learned_variance=learned_variance,
                                                                          lowres_noise_level=lowres_noise_level)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(
            self,
            unet,
            shape,
            image_embed,
            noise_scheduler,
            predict_x_start=False,
            predict_v=False,
            learned_variance=False,
            clip_denoised=True,
            lowres_cond_img=None,
            text_encodings=None,
            cond_scale=1,
            is_latent_diffusion=False,
            lowres_noise_level=None,
            inpaint_image=None,
            inpaint_mask=None,
            inpaint_resample_times=5
    ):
        device = self.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        is_inpaint = exists(inpaint_image)
        resample_times = inpaint_resample_times if is_inpaint else 1

        if is_inpaint:
            inpaint_image = self.normalize_img(inpaint_image)
            inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest=True)
            inpaint_mask = rearrange(inpaint_mask, 'b h w -> b 1 h w').float()
            inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest=True)
            inpaint_mask = inpaint_mask.bool()

        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        for time in tqdm(reversed(range(0, noise_scheduler.num_timesteps)), desc='sampling loop time step',
                         total=noise_scheduler.num_timesteps):
            is_last_timestep = time == 0

            for r in reversed(range(0, resample_times)):
                is_last_resample_step = r == 0

                times = torch.full((b,), time, device=device, dtype=torch.long)

                if is_inpaint:
                    # following the repaint paper
                    # https://arxiv.org/abs/2201.09865
                    noised_inpaint_image = noise_scheduler.q_sample(inpaint_image, t=times)
                    img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

                self_cond = x_start if unet.self_cond else None

                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    image_embed=image_embed,
                    text_encodings=text_encodings,
                    cond_scale=cond_scale,
                    self_cond=self_cond,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_level=lowres_noise_level,
                    predict_x_start=predict_x_start,
                    predict_v=predict_v,
                    noise_scheduler=noise_scheduler,
                    learned_variance=learned_variance,
                    clip_denoised=clip_denoised
                )

                if is_inpaint and not (is_last_timestep or is_last_resample_step):
                    # in repaint, you renoise and resample up to 10 times every step
                    img = noise_scheduler.q_sample_from_to(img, times - 1, times)

        if is_inpaint:
            img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    def p_sample_loop_ddim(
            self,
            unet,
            shape,
            image_embed,
            noise_scheduler,
            timesteps,
            eta=1.,
            predict_x_start=False,
            predict_v=False,
            learned_variance=False,
            clip_denoised=True,
            lowres_cond_img=None,
            text_encodings=None,
            cond_scale=1,
            is_latent_diffusion=False,
            lowres_noise_level=None,
            inpaint_image=None,
            inpaint_mask=None,
            inpaint_resample_times=5
    ):
        batch, device, total_timesteps, alphas, eta = shape[
            0], self.device, noise_scheduler.num_timesteps, noise_scheduler.alphas_cumprod, self.ddim_sampling_eta

        times = torch.linspace(0., total_timesteps, steps=timesteps + 2)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))

        is_inpaint = exists(inpaint_image)
        resample_times = inpaint_resample_times if is_inpaint else 1

        if is_inpaint:
            inpaint_image = self.normalize_img(inpaint_image)
            inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest=True)
            inpaint_mask = rearrange(inpaint_mask, 'b h w -> b 1 h w').float()
            inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest=True)
            inpaint_mask = inpaint_mask.bool()

        img = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            is_last_timestep = time_next == 0

            for r in reversed(range(0, resample_times)):
                is_last_resample_step = r == 0

                alpha = alphas[time]
                alpha_next = alphas[time_next]

                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

                if is_inpaint:
                    # following the repaint paper
                    # https://arxiv.org/abs/2201.09865
                    noised_inpaint_image = noise_scheduler.q_sample(inpaint_image, t=time_cond)
                    img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

                self_cond = x_start if unet.self_cond else None

                unet_output = unet.forward_with_cond_scale(img, time_cond, image_embed=image_embed,
                                                           text_encodings=text_encodings, cond_scale=cond_scale,
                                                           self_cond=self_cond, lowres_cond_img=lowres_cond_img,
                                                           lowres_noise_level=lowres_noise_level)

                pred, _ = self.parse_unet_output(learned_variance, unet_output)

                # predict x0

                if predict_v:
                    x_start = noise_scheduler.predict_start_from_v(img, t=time_cond, v=pred)
                elif predict_x_start:
                    x_start = pred
                else:
                    x_start = noise_scheduler.predict_start_from_noise(img, t=time_cond, noise=pred)

                # maybe clip x0

                if clip_denoised:
                    x_start = self.dynamic_threshold(x_start)

                # predict noise

                pred_noise = noise_scheduler.predict_noise_from_start(img, t=time_cond, x0=x_start)

                c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
                noise = torch.randn_like(img) if not is_last_timestep else 0.

                img = x_start * alpha_next.sqrt() + \
                      c1 * noise + \
                      c2 * pred_noise

                if is_inpaint and not (is_last_timestep or is_last_resample_step):
                    # in repaint, you renoise and resample up to 10 times every step
                    time_next_cond = torch.full((batch,), time_next, device=device, dtype=torch.long)
                    img = noise_scheduler.q_sample_from_to(img, time_next_cond, time_cond)

        if exists(inpaint_image):
            img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        img = self.unnormalize_img(img)
        return img

    @torch.no_grad()
    def p_sample_loop(self, *args, noise_scheduler, timesteps=None, **kwargs):
        num_timesteps = noise_scheduler.num_timesteps

        timesteps = default(timesteps, num_timesteps)
        assert timesteps <= num_timesteps
        is_ddim = timesteps < num_timesteps

        if not is_ddim:
            return self.p_sample_loop_ddpm(*args, noise_scheduler=noise_scheduler, **kwargs)

        return self.p_sample_loop_ddim(*args, noise_scheduler=noise_scheduler, timesteps=timesteps, **kwargs)

    def p_losses(self, unet, x_start, times, *, image_embed, noise_scheduler, lowres_cond_img=None, text_encodings=None,
                 predict_x_start=False, predict_v=False, noise=None, learned_variance=False, clip_denoised=False,
                 is_latent_diffusion=False, lowres_noise_level=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]
        if not is_latent_diffusion:
            x_start = self.normalize_img(x_start)
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t
        x_noisy = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)

        # unet kwargs
        unet_kwargs = dict(
            image_embed=image_embed,
            text_encodings=text_encodings,
            lowres_cond_img=lowres_cond_img,
            lowres_noise_level=lowres_noise_level,
        )

        # self conditioning
        self_cond = None
        if unet.self_cond and random.random() < 0.5:
            with torch.no_grad():
                unet_output = unet(x_noisy, times, **unet_kwargs)
                self_cond, _ = self.parse_unet_output(learned_variance, unet_output)
                self_cond = self_cond.detach()

        # forward to get model prediction
        unet_output = unet(
            x_noisy,
            times,
            **unet_kwargs,
            self_cond=self_cond,
            image_cond_drop_prob=self.image_cond_drop_prob,
            text_cond_drop_prob=self.text_cond_drop_prob,
        )

        pred, _ = self.parse_unet_output(learned_variance, unet_output)

        if predict_v:
            target = noise_scheduler.calculate_v(x_start, times, noise)
        elif predict_x_start:
            target = x_start
        else:
            target = noise

        loss = noise_scheduler.loss_fn(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = noise_scheduler.p2_reweigh_loss(loss, times)
        loss = loss.mean()

        if not learned_variance:
            # return simple loss if not using learned variance
            return loss

        # most of the code below is transcribed from
        # https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
        # the Improved DDPM paper then further modified it so that the mean is detached (shown a couple lines before), and weighted to be smaller than the l1 or l2 "simple" loss
        # it is questionable whether this is really needed, looking at some of the figures in the paper, but may as well stay faithful to their implementation

        # if learning the variance, also include the extra weight kl loss

        true_mean, _, true_log_variance_clipped = noise_scheduler.q_posterior(x_start=x_start, x_t=x_noisy, t=times)
        model_mean, _, model_log_variance, _ = self.p_mean_variance(unet, x=x_noisy, t=times, image_embed=image_embed,
                                                                    noise_scheduler=noise_scheduler,
                                                                    clip_denoised=clip_denoised, learned_variance=True,
                                                                    model_output=unet_output)

        # kl loss with detached model predicted mean, for stability reasons as in paper

        detached_model_mean = model_mean.detach()

        kl = normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = meanflat(kl) * NAT

        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=detached_model_mean,
                                                           log_scales=0.5 * model_log_variance)
        decoder_nll = meanflat(decoder_nll) * NAT

        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        vb_losses = torch.where(times == 0, decoder_nll, kl)

        # weight the vb loss smaller, for stability, as in the paper (recommended 0.001)

        vb_loss = vb_losses.mean() * self.vb_loss_weight

        return loss + vb_loss

    @torch.no_grad()
    @eval_decorator
    def sample(
            self,
            image=None,
            image_embed=None,
            text=None,
            text_encodings=None,
            batch_size=1,
            cond_scale=1.,
            start_at_unet_number=1,
            stop_at_unet_number=None,
            distributed=False,
            inpaint_image=None,
            inpaint_mask=None,
            inpaint_resample_times=5,
            one_unet_in_gpu_at_time=True
    ):
        assert self.unconditional or exists(
            image_embed), 'image embed must be present on sampling from decoder unless if trained unconditionally'

        if not self.unconditional:
            batch_size = image_embed.shape[0]

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip)
            _, text_encodings = self.clip.embed_text(text)

        assert not (self.condition_on_text_encodings and not exists(
            text_encodings)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text_encodings and exists(
            text_encodings)), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (exists(inpaint_image) ^ exists(
            inpaint_mask)), 'inpaint_image and inpaint_mask (boolean mask of [batch, height, width]) must be both given for inpainting'

        img = None
        if start_at_unet_number > 1:
            # Then we are not generating the first image and one must have been passed in
            assert exists(image), 'image must be passed in if starting at unet number > 1'
            assert image.shape[
                       0] == batch_size, 'image must have batch size of {} if starting at unet number > 1'.format(
                batch_size)
            prev_unet_output_size = self.image_sizes[start_at_unet_number - 2]
            img = resize_image_to(image, prev_unet_output_size, nearest=True)

        is_cuda = next(self.parameters()).is_cuda

        num_unets = self.num_unets
        cond_scale = cast_tuple(cond_scale, num_unets)

        for unet_number, unet, vae, channel, image_size, predict_x_start, predict_v, learned_variance, noise_scheduler, lowres_cond, sample_timesteps, unet_cond_scale in tqdm(
                zip(range(1, num_unets + 1), self.unets, self.vaes, self.sample_channels, self.image_sizes,
                    self.predict_x_start, self.predict_v, self.learned_variance, self.noise_schedulers,
                    self.lowres_conds, self.sample_timesteps, cond_scale)):
            if unet_number < start_at_unet_number:
                continue  # It's the easiest way to do it

            context = self.one_unet_in_gpu(unet=unet) if is_cuda and one_unet_in_gpu_at_time else null_context()

            with context:
                # prepare low resolution conditioning for upsamplers
                lowres_cond_img = lowres_noise_level = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_cond_img = resize_image_to(img, target_image_size=image_size,
                                                      clamp_range=self.input_image_range, nearest=True)

                    if lowres_cond.use_noise:
                        lowres_noise_level = torch.full((batch_size,), int(self.lowres_noise_sample_level * 1000),
                                                        dtype=torch.long, device=self.device)
                        lowres_cond_img, _ = lowres_cond.noise_image(lowres_cond_img, lowres_noise_level)

                # latent diffusion

                is_latent_diffusion = isinstance(vae, VQGanVAE)
                image_size = vae.get_encoded_fmap_size(image_size)
                shape = (batch_size, vae.encoded_dim, image_size, image_size)

                lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

                # denoising loop for image

                img = self.p_sample_loop(
                    unet,
                    shape,
                    image_embed=image_embed,
                    text_encodings=text_encodings,
                    cond_scale=unet_cond_scale,
                    predict_x_start=predict_x_start,
                    predict_v=predict_v,
                    learned_variance=learned_variance,
                    clip_denoised=not is_latent_diffusion,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_level=lowres_noise_level,
                    is_latent_diffusion=is_latent_diffusion,
                    noise_scheduler=noise_scheduler,
                    timesteps=sample_timesteps,
                    inpaint_image=inpaint_image,
                    inpaint_mask=inpaint_mask,
                    inpaint_resample_times=inpaint_resample_times
                )

                img = vae.decode(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        return img

    def forward(
            self,
            image,
            text=None,
            image_embed=None,
            text_encodings=None,
            unet_number=None,
            return_lowres_cond_image=False
            # whether to return the low resolution conditioning images, for debugging upsampler purposes
    ):
        assert not (self.num_unets > 1 and not exists(
            unet_number)), f'you must specify which unet you want trained, from a range of 1 to {self.num_unets}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1

        unet = self.get_unet(unet_number)

        vae = self.vaes[unet_index]
        noise_scheduler = self.noise_schedulers[unet_index]
        lowres_conditioner = self.lowres_conds[unet_index]
        target_image_size = self.image_sizes[unet_index]
        predict_x_start = self.predict_x_start[unet_index]
        predict_v = self.predict_v[unet_index]
        random_crop_size = self.random_crop_sizes[unet_index]
        learned_variance = self.learned_variance[unet_index]
        b, c, h, w, device, = *image.shape, image.device

        assert image.shape[1] == self.channels
        assert h >= target_image_size and w >= target_image_size

        times = torch.randint(0, noise_scheduler.num_timesteps, (b,), device=device, dtype=torch.long)

        if not exists(image_embed) and not self.unconditional:
            assert exists(
                self.clip), 'if you want to derive CLIP image embeddings automatically, you must supply `clip` to the decoder on init'
            image_embed, _ = self.clip.embed_image(image)

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip), 'if you are passing in raw text, you need to supply `clip` to the decoder'
            _, text_encodings = self.clip.embed_text(text)

        assert not (self.condition_on_text_encodings and not exists(
            text_encodings)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text_encodings and exists(
            text_encodings)), 'decoder specified not to be conditioned on text, yet it is presented'

        lowres_cond_img, lowres_noise_level = lowres_conditioner(image, target_image_size=target_image_size,
                                                                 downsample_image_size=self.image_sizes[
                                                                     unet_index - 1]) if exists(
            lowres_conditioner) else (None, None)
        image = resize_image_to(image, target_image_size, nearest=True)

        if exists(random_crop_size):
            aug = K.RandomCrop((random_crop_size, random_crop_size), p=1.)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            image = aug(image)
            lowres_cond_img = aug(lowres_cond_img, params=aug._params)

        is_latent_diffusion = not isinstance(vae, NullVQGanVAE)

        vae.eval()
        with torch.no_grad():
            image = vae.encode(image)
            lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

        # print('UNet: ', unet_number, ', lowres_noise_level is None: ', lowres_noise_level is None, ', lowres_cond_img is None: ', lowres_cond_img is None)

        losses = self.p_losses(unet, image, times, image_embed=image_embed, text_encodings=text_encodings,
                               lowres_cond_img=lowres_cond_img, predict_x_start=predict_x_start, predict_v=predict_v,
                               learned_variance=learned_variance, is_latent_diffusion=is_latent_diffusion,
                               noise_scheduler=noise_scheduler, lowres_noise_level=lowres_noise_level)

        if not return_lowres_cond_image:
            return losses

        return losses, lowres_cond_img
