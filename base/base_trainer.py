import torch.cuda
import torch.nn as nn
from torch.optim import *
import yaml
from abc import abstractmethod
from models.model import *
from models.loss import *
from models.metrics import *
from torch.optim.lr_scheduler import *
from data_loader.FaceLoader import FaceDataLoader
import math


class BaseTrainer:
    def __init__(self, hyp, device=None, logger=None):
        self.logger = logger
        self.hyp = hyp
        self.device = (
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        m = hyp["arch"]["type"]
        self.model = eval(m) if isinstance(m, str) else m
        self.model = self.model(**hyp["arch"]["args"])

        dl = hyp["data_loader"]
        dalo = dl.get("type")
        if dalo:
            train_ = dl.get("train", False)
            test_ = dl.get("test", False)
            train_test = dl.get("train_val", False)
            assert not ((train_ and test_) and train_test), print("Bad format!")
            if train_:
                if train_.get("augment"):
                    train_["hyp"] = hyp["hyp"]
                self.train_loader = eval(dalo)(**train_)
            if test_:
                self.test_loader = eval(dalo)(**test_)
            if train_test:
                if train_test.get("augment"):
                    train_test["hyp"] = hyp["hyp"]
                self.train_loader, self.test_loader = eval(dalo)(
                    **train_test
                ).get_loader()

        if hyp.get("optimizer") is not None:
            opt = hyp["optimizer"]
            self.optimizer = eval(opt["type"])
            self.optimizer = self.optimizer(self.model.parameters(), **opt["args"])
        else:
            self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        loss = hyp["loss"]
        loss_args = loss.get("args") if isinstance(loss, dict) else None
        if loss_args:
            loss = loss["type"]
            self.loss = eval(loss)(**loss_args)
        else:
            self.loss = eval(loss)()
        metrics = hyp.get("metrics")
        if metrics is not None:
            self.metrics = eval(metrics)
        train_para = hyp["trainer"]
        self.epochs = train_para.get("epoch", 50)
        self.tensorboard = train_para.get("tensorboard", False)
        self.hyparameters = hyp.get("hyp")
        self.model.hyp = self.hyparameters
        self.name = self.hyp.get("name", type(self.model).__name__)
        del train_para, opt, loss, loss_args, m, metrics, train_, test_, dl, dalo

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        pass
