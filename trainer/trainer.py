import logging
import random
import time
import sys
import numpy as np
import torch.cuda
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import yaml
import math
from torch.optim import *
from base.base_trainer import BaseTrainer
from models.metrics import *
from utils.autoanchor import check_anchors
from utils.general import init_seeds, increment_path, fitness
from utils.torch_utils import intersect_dicts, ModelEMA
import torch.nn.functional as F
from PIL import ImageFile
from torch.optim.lr_scheduler import *

sys.path.append("./")
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Trainer(BaseTrainer):
    def __init__(
        self,
        hyp,
        device,
        logger=None,
        pretrained=None,
        save_dir="runs/train",
        use_ema=True,
        multi_scale=False,
        resume=False,
    ):
        super(Trainer, self).__init__(hyp, device, logger)
        self.resume = resume
        self.nb = len(self.train_loader)
        self.gs = int(max(self.model.stride))
        self.nc = self.model.nc
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model.nc = self.nc
        else:
            self.model = self.model.to(self.device)
        self.warmup_iter = None
        self.use_ema = use_ema
        self.batch_size = self.train_loader.batch_size
        self.ema = ModelEMA(self.model) if self.use_ema else None
        self.accumulate = max(
            round(64 / self.batch_size), 1
        )  # accumulate loss before optimizing
        self.multi_scale = multi_scale
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device != "cpu"))
        self.pretrained = pretrained

        self.save_dir = self.save_dir = Path(
            increment_path(Path(save_dir) / self.name, exist_ok=False)
        )
        opti = hyp["optimizer"]
        self.optimizer = eval(opti["type"])
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        weight_decay = (
            opti["args"].pop("weight_decay")
            if "weight_decay" in opti["args"].keys()
            else 0
        )
        self.optimizer = self.optimizer(pg0, **opti["args"])
        self.optimizer.add_param_group(
            {"params": pg1, "weight_decay": weight_decay}
        )  # add pg1 with weight_decay
        self.optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
        lr_scheduler_ = hyp.get("lr_scheduler", None)
        if lr_scheduler_:
            self.lr_scheduler = eval(lr_scheduler_["type"])
            args = lr_scheduler_["args"]
            if self.lr_scheduler is LambdaLR:
                func = args.pop("lr_lambda")
                self.lf = lambda x: eval(
                    func, {"epochs": self.epochs, "math": math, "x": x}
                )
                self.lr_scheduler = self.lr_scheduler(
                    self.optimizer, lr_lambda=self.lf, **args
                )
            else:
                self.lr_scheduler = self.lr_scheduler(self.optimizer, **args)
        if self.hyp["data_loader"].get("train", False) or self.hyp["data_loader"].get(
            "test", False
        ):
            imgsz, imgsz_test = [
                self.hyp["data_loader"]["train"].get("imgsz", False),
                self.hyp["data_loader"]["test"].get("imgsz", False),
            ]
            if not (imgsz and imgsz_test):
                imgsz = imgsz_test = 640
            elif imgsz and not imgsz_test:
                imgsz_test = imgsz
            elif not imgsz and imgsz_test:
                imgsz = imgsz_test
        else:
            imgsz = imgsz_test = self.hyp["data_loader"]["train_val"].get("imgsz")

        self.imgsz, self.imgsz_test = [
            check_img_size(x, self.gs) for x in [imgsz, imgsz_test]
        ]

    def _train_epoch(self, epoch):
        mloss = torch.zeros(4, device=self.device)
        pdar = enumerate(self.train_loader)
        self.logger.info(
            ("\n" + "%10s" * 8)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "targets", "img_size")
        ) if self.logger else print(
            ("\n" + "%10s" * 8)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "targets", "img_size")
        )

        pdar = tqdm(pdar, total=self.nb)
        self.optimizer.zero_grad()
        s = ""
        # i, (imgs, targets, paths, _) = 0, next(iter(self.train_loader))
        for i, (imgs, targets, paths, _) in pdar:
            ni = i + self.nb * epoch
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0

            # Warmup
            if ni < self.warmup_iter:
                xi = [0, self.warmup_iter]
                self.accumulate = max(
                    1, np.interp(ni, xi, [1, 64 / self.batch_size]).round()
                )
                for j, x in enumerate(self.optimizer.param_groups):
                    x["lr"] = np.interp(
                        ni,
                        xi,
                        [
                            self.hyparameters["warmup_bias_lr"] if j == 2 else 0.0,
                            x["initial_lr"] * self.lf(epoch),
                        ],
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni,
                            xi,
                            [
                                self.hyparameters["warmup_momentum"],
                                self.hyparameters["momentum"],
                            ],
                        )

            if self.multi_scale:
                sz = (
                    random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs)
                    // self.gs
                    * self.gs
                )
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]
                    imgs = F.interpolate(
                        imgs, size=ns, mode="bilinear", align_corners=False
                    )

            # Forward
            with torch.cuda.amp.autocast(enabled=(self.device != "cpu")):
                pred = self.model(imgs)
                loss, loss_item = self.loss(pred, targets.to(self.device), self.model)

            # Backward
            self.scaler.scale(loss).backward()

            # Optimize
            if ni % self.accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.use_ema:
                    self.ema.update(self.model)

            mloss = (mloss * i + loss_item) / (i + 1)
            mem = (
                "%.3gG" % (torch.cuda.memory_reserved() / 1e9)
                if torch.cuda.is_available()
                else 0
            )
            s = ("%10s" * 2 + "%10.4g" * 6) % (
                "%g/%g" % (epoch, self.epochs - 1),
                mem,
                *mloss,
                targets.shape[0],
                imgs.shape[-1],
            )
            pdar.set_description(s)

            # end batch -----------------------------------------------------------------
        # end epoch ----------------------------------------------------------

        self.lr_scheduler.step()
        return mloss, s

    def train(self):
        tb_writer = SummaryWriter(self.save_dir) if self.tensorboard else None
        wdir = self.save_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        best = wdir / f"{self.name}-best.pt"
        result_file = self.save_dir / "result.txt"
        init_seeds(11 + 11 + 2001)

        pretrained = (
            self.pretrained.endswith(".pt")
            or self.pretrained.endswith(".pth")
            or self.pretrained.endswith(".taz")
            if self.pretrained
            else False
        )
        ckpt = None
        if pretrained:
            print("Loading pretrained....")
            ckpt = torch.load(self.pretrained, map_location=self.device)
            state_dict = ckpt.get("model") if ckpt.get("model") else ckpt["state_dict"]
            state_dict = intersect_dicts(state_dict, self.model.state_dict())
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(
                f"Transferred {len(state_dict)}/{len(self.model.state_dict())} items from {self.pretrained}"
            ) if self.logger else print()

        start_epoch, best_fitness = 0, 0.0
        if pretrained and self.resume:
            if ckpt.get("optimizer") is not None:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                best_fitness = ckpt["best_fitness"]

            if ckpt.get("training_results") is not None:
                with open(result_file, "w") as file:
                    file.write(ckpt["training_results"])
            start_epoch = ckpt["epoch"] + 1
            assert start_epoch < self.epochs, ""
            print("Resuming training, start epoch: ", start_epoch)
            del ckpt, state_dict
        if self.use_ema:
            self.ema.updates = start_epoch * self.nb // self.accumulate
        train_dataset = self.train_loader.get_dataset()
        anchor_t = (
            self.hyparameters.get("anchor_t")
            if self.hyparameters.get("anchor_t")
            else 4.0
        )
        check_anchors(train_dataset, model=self.model, thr=anchor_t, imgsz=self.imgsz)
        self.hyparameters["cls"] *= (
            self.nc / 80
        )  # scale class loss gain to current dataset (by default 0.5 in coco)
        self.model.hyp = self.hyparameters
        self.model.gr = 1.0  # iou loss ratio (obj loss = 1.0 or iou)
        self.model.name = self.name
        self.warmup_iter = max(round(self.hyparameters["warmup_epoch"] * self.nb), 1000)
        results = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.lr_scheduler.last_epoch = start_epoch - 1

        self.logger.info(
            "Image sizes %g train, %g test\n"
            "Using %g dataloader workers\nLogging results to %s\n"
            "Starting training for %g epochs..."
            % (
                self.imgsz,
                self.imgsz_test,
                self.train_loader.num_workers,
                self.save_dir,
                self.epochs,
            )
        ) if self.logger else print()
        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            mloss, s = self._train_epoch(epoch)
            # mloss, s = [0] * 4, ''
            lr = [x["lr"] for x in self.optimizer.param_groups]
            if (epoch > 15) or (epoch == self.epochs - 1) or epoch == 0:
                if self.use_ema:
                    self.ema.update_attr(
                        self.model, include=["nc", "name", "stride", "gr", "hyp"]
                    )

                results, maps, times = self.metrics(
                    self.test_loader,
                    self.ema.ema if self.use_ema else self.model,
                    batch_size=self.batch_size,
                    imgsz=self.imgsz_test,
                    save_dir=self.save_dir,
                )
                with open(result_file, "a") as f:
                    f.write(s + "%10.4g" * 7 % results + "\n")

                tags = [
                    "train/box_loss",
                    "train/obj_loss",
                    "train/cls_loss",  # train loss
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",  # val loss
                    "x/lr0",
                    "x/lr1",
                    "x/lr2",
                ]
                for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                    if tb_writer:
                        tb_writer.add_scalar(tag, x, epoch)

                fi = fitness(np.array(results).reshape(1, -1))
                if fi > best_fitness:
                    best_fitness = fi
                    with open(result_file, "r") as f:
                        ckpt = {
                            "epoch": epoch,
                            "best_fitness": best_fitness,
                            "training_results": f.read(),
                            "ema": self.ema.ema.state_dict() if self.use_ema else None,
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                        }
                    torch.save(ckpt, best)

                    del ckpt
        # end training ------------------
        torch.cuda.empty_cache()
        return results
