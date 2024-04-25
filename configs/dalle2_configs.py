from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any, Union, TypeVar, Tuple
import json
InnerType = TypeVar('InnerType')
ListOrTuple = Union[List[InnerType], Tuple[InnerType]]
SingularOrIterable = Union[InnerType, ListOrTuple[InnerType]]


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class AdapterConfig(BaseModel):
    make: str = "openai"
    model: str = "ViT-L/14"
    base_model_kwargs: Optional[Dict[str, Any]] = None


class DiffusionPriorNetworkConfig(BaseModel):
    dim: int
    depth: int
    max_text_len: Optional[int] = None
    num_timesteps: Optional[int] = None
    num_time_embeds: int = 1
    num_image_embeds: int = 1
    num_text_embeds: int = 1
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    norm_in: bool = False
    norm_out: bool = True
    attn_dropout: float = 0.
    ff_dropout: float = 0.
    final_proj: bool = True
    normformer: bool = False
    rotary_emb: bool = True

    class Config:
        extra = "allow"


class DiffusionPriorConfig(BaseModel):
    clip: Optional[AdapterConfig] = None
    net: DiffusionPriorNetworkConfig
    image_embed_dim: int
    image_size: int
    image_channels: int = 3
    timesteps: int = 1000
    sample_timesteps: Optional[int] = None
    cond_drop_prob: float = 0.
    loss_type: str = 'l2'
    predict_x_start: bool = True
    beta_schedule: str = 'cosine'
    condition_on_text_encodings: bool = True

    class Config:
        extra = "allow"


class DiffusionPriorTrainConfig(BaseModel):
    epochs: int = 1
    batch_size: int = 1
    lr: float = 1.1e-4
    wd: float = 6.02e-2
    max_grad_norm: float = 0.5
    use_ema: bool = True
    ema_beta: float = 0.99
    amp: bool = False
    warmup_steps: Optional[int] = None   # number of warmup steps
    save_every_seconds: int = 3600       # how often to save
    eval_timesteps: List[int] = [64]     # which sampling timesteps to evaluate with
    best_validation_loss: float = 1e9    # the current best valudation loss observed
    current_epoch: int = 0               # the current epoch
    num_samples_seen: int = 0            # the current number of samples seen
    random_seed: int = 0                 # manual seed for torch
    train_img_path: str = ''
    train_annot_path: str = ''


class TrainDiffusionPriorConfig(BaseModel):
    prior: DiffusionPriorConfig
    train: DiffusionPriorTrainConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)


class UnetConfig(BaseModel):
    dim: int
    dim_mults: ListOrTuple[int]
    image_embed_dim: Optional[int] = None
    text_embed_dim: Optional[int] = None
    cond_on_text_encodings: Optional[bool] = None
    cond_dim: Optional[int] = None
    channels: int = 3
    self_attn: SingularOrIterable[bool] = False
    attn_dim_head: int = 32
    attn_heads: int = 16
    init_cross_embed: bool = True

    class Config:
        extra = "allow"


class DecoderConfig(BaseModel):
    unets: ListOrTuple[UnetConfig]
    image_size: Optional[int] = None
    image_sizes: ListOrTuple[int] = None
    clip: Optional[AdapterConfig] = None   # The clip model to use if embeddings are not provided
    channels: int = 3
    timesteps: int = 1000
    sample_timesteps: Optional[SingularOrIterable[Optional[int]]] = None
    loss_type: str = 'l2'
    beta_schedule: str = None  # None means all cosine
    learned_variance: SingularOrIterable[bool] = True
    image_cond_drop_prob: float = 0.1
    text_cond_drop_prob: float = 0.5

    @validator('image_sizes')
    def check_image_sizes(cls, image_sizes, values):
        if exists(values.get('image_size')) ^ exists(image_sizes):
            return image_sizes
        raise ValueError('either image_size or image_sizes is required, but not both')

    class Config:
        extra = "allow"


class DecoderTrainConfig(BaseModel):
    epochs: int = 20
    batch_size: int = 1
    lr: SingularOrIterable[float] = 1e-4
    wd: SingularOrIterable[float] = 0.01
    warmup_steps: Optional[SingularOrIterable[int]] = None
    find_unused_parameters: bool = True
    static_graph: bool = True
    max_grad_norm: SingularOrIterable[float] = 0.5
    save_every_n_samples: int = 100000
    n_sample_images: int = 6                       # The number of example images to produce when sampling the train and test dataset
    cond_scale: Union[float, List[float]] = 1.0
    device: str = 'cuda:0'
    epoch_samples: Optional[int] = None                      # Limits the number of samples per epoch. None means no limit. Required if resample_train is true as otherwise the number of samples per epoch is infinite.
    validation_samples: Optional[int] = None                 # Same as above but for validation.
    save_immediately: bool = False
    use_ema: bool = True
    ema_beta: float = 0.999
    amp: bool = False
    unet_training_mask: Optional[ListOrTuple[bool]] = None   # If None, use all unets
    train_img_path: str = ''
    train_annot_path: str = ''


class TrainDecoderConfig(BaseModel):
    decoder: DecoderConfig
    train: DecoderTrainConfig
    seed: int = 0

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
            print(config)
        return cls(**config)