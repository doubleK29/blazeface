import logging

from PIL import ExifTags
from torch.utils.data.dataloader import DataLoader

# Parameters
img_formats = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
]  # acceptable image suffixes
vid_formats = [
    "mov",
    "avi",
    "mp4",
    "mpg",
    "mpeg",
    "m4v",
    "wmv",
    "mkv",
]  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


class BaseDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size, collate_fn=None, workers=8, pin_memory=True
    ):
        self.batch_size = min(batch_size, len(dataset))
        self.dataset = dataset
        super(BaseDataLoader, self).__init__(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    def get_dataset(self):
        return self.dataset
