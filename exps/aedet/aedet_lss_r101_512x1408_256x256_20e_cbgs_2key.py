"""
mAP: 0.4488
mATE: 0.5289
mASE: 0.2623
mAOE: 0.3189
mAVE: 0.3157
mAAE: 0.2034
NDS: 0.5615
Eval time: 85.8s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.645	0.380	0.147	0.060	0.304	0.211
truck	0.363	0.573	0.192	0.064	0.277	0.182
bus	0.508	0.552	0.187	0.075	0.521	0.231
trailer	0.243	0.866	0.227	0.271	0.249	0.216
construction_vehicle	0.085	0.867	0.471	0.870	0.117	0.370
pedestrian	0.481	0.540	0.288	0.556	0.368	0.184
motorcycle	0.463	0.483	0.250	0.356	0.515	0.214
bicycle	0.430	0.409	0.267	0.503	0.175	0.019
traffic_cone	0.650	0.309	0.311	nan	nan	nan
barrier	0.621	0.311	0.282	0.115	nan	nan
"""
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from callbacks.ema import EMACallback
from exps.aedet.aedet_lss_r101_512x1408_256x256_24e_2key import \
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
            self.batch_size_per_device * self.gpus * self.num_nodes
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-1)
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

    parent_parser.add_argument("--local_rank", default=-1, type=int)
    parent_parser.add_argument("--gpu_count", type=int, default=1, help="")
    parent_parser.add_argument('--dist_url', type=str, default="")

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
                        accelerator='gpu',  # ddp
                        num_sanity_val_steps=0,
                        gradient_clip_val=5,
                        limit_val_batches=1.0,
                        check_val_every_n_epoch=30,
                        enable_checkpointing=False,
                        precision=16,
                        default_root_dir='./outputs/aedet_lss_r101_512x1408_256x256_20e_cbgs_2key',
                        devices=8,
                        strategy="ddp"
                        )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
