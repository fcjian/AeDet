"""
AP: 0.3809
mATE: 0.5888
mASE: 0.2762
mAOE: 0.4426
mAVE: 0.3705
mAAE: 0.2059
NDS: 0.5020
Eval time: 104.7s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.579	0.469	0.157	0.095	0.361	0.211
truck	0.308	0.626	0.207	0.096	0.313	0.201
bus	0.432	0.580	0.201	0.077	0.646	0.245
trailer	0.199	0.819	0.267	0.317	0.234	0.187
construction_vehicle	0.089	0.821	0.490	1.075	0.099	0.351
pedestrian	0.368	0.664	0.296	0.790	0.478	0.241
motorcycle	0.379	0.588	0.256	0.525	0.630	0.192
bicycle	0.327	0.497	0.273	0.885	0.204	0.019
traffic_cone	0.555	0.413	0.329	nan	nan	nan
barrier	0.574	0.411	0.287	0.124	nan	nan
"""
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from callbacks.ema import EMACallback
from exps.aedet.aedet_lss_r50_256x704_128x128_24e_2key import \
    AeDetLightningModel as BaseAeDetLightningModel
from layers.backbones.lss_fpn import LSSFPN as BaseLSSFPN
from layers.heads.aedet_head import AeDetHead
from models.aedet import AeDet


class AeDetLightningModel(BaseAeDetLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = AeDet(self.backbone_conf,
                              self.head_conf,
                              is_train_depth=True)
        self.data_use_cbgs = True

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=2e-1)
        return [optimizer]


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = AeDetLightningModel(**vars(args))
    train_dataloader = model.train_dataloader()

    if args.ckpt_path:
        ema_callback = EMACallback(len(train_dataloader.dataset) * args.max_epochs, ema_ckpt_path=args.ckpt_path.replace('origin', 'ema'))
    else:
        ema_callback = EMACallback(len(train_dataloader.dataset) * args.max_epochs)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback])
    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, ckpt_path=args.ckpt_path)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = AeDetLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=20,
                        accelerator='ddp',
                        num_sanity_val_steps=0,
                        gradient_clip_val=5,
                        limit_val_batches=1.0,
                        check_val_every_n_epoch=4,
                        enable_checkpointing=False,
                        precision=16,
                        default_root_dir='./outputs/aedet_lss_r50_256x704_128x128_20e_cbgs_2key')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
