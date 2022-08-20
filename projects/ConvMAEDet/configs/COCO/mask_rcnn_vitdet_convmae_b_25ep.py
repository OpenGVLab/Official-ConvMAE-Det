from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler
import torch.nn as nn

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from projects.ConvMAEDet.modeling.convmae import get_vit_lr_decay_rate
from projects.ConvMAEDet.modeling.convmae import ConvViT

from ..common.coco_loader_lsj import dataloader


model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

model.backbone.net = L(ConvViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=(4, 2, 2),
        embed_dim=(256, 384, 768),
        depth=(2, 2, 11),
        num_heads=12,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 1, 4, 7, 10 for global attention
            0,
            2,
            3,
            5,
            6,
            8,
            9,
        ],
        use_rel_pos=True,
        pretrain_use_cls_token=False,
        out_feature="last_feat",
    )

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = ""
train.output_dir = "./output/vitdet_convmae_base_mask_rcnn_25ep_LSJ"
train.checkpointer = dict(period=2500, max_to_keep=2)
train.eval_period = 7500


# Schedule
# 25 ep = 184375 iters * 16 images/iter / 118000 images/ep
train.max_iter = 184375

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=11, lr_decay_rate=0.8)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

dataloader.train.total_batch_size = 16
