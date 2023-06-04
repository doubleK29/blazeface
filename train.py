import logging
import argparse
import yaml

from logger.logger import set_logging
from trainer.trainer import Trainer
from utils.torch_utils import select_device

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info(
        "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)"
    )

if __name__ == "__main__":
    set_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    opt = parser.parse_args()
    # device = f"cuda:{int(opt.device)}" if opt.device.isnumeric() else "cpu"
    device = select_device(opt.device, batch_size=64)
    hyp = yaml.load(open("config/hyp.trainer_blazeface.yaml"), Loader=yaml.FullLoader)
    trainer_ = Trainer(
        hyp,
        device,
        logger,
        pretrained="runs/train/YOLO_exp12/weights/YOLO_exp-best.pt",
        resume=True,
        use_ema=False,
    )
    trainer_.train()
