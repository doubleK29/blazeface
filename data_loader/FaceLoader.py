import logging
from itertools import repeat
from multiprocessing.pool import ThreadPool

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import json
from data_loader.augmentation import *
from base.base_data_loader import BaseDataLoader, orientation
from utils.general import xyxy2xywh, get_hash


class LoadFaceImagesAndLabels(Dataset):  # for training/testing
    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        square=False,
        stride=32,
        pad=0.0,
    ):
        self.img_size = img_size
        self.augment = augment
        self.square = square
        self.hyp = hyp
        self.rect = rect
        self.mosaic = (
            self.augment and not self.rect and not self.square
        )  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        json_path = [p[:-1] for p in open(path, "r").readlines()]
        try:
            f = []  # image files
            for p in tqdm(json_path, desc="Finding label files"):
                image_path = p.replace(".json", ".jpg")
                p = Path(p)  # os-agnostic
                if p.is_file() and Path(image_path).is_file():  # file
                    f.append(image_path)
                else:
                    print(f"WARMING: No path {p}")
            self.img_files = f
            assert self.img_files, "No images found"
        except Exception as e:
            raise Exception("Error loading data from %s: %s \n" % (path, e))

        # Check cache
        self.label_files = json_path  # labels
        cache_path = Path(path).parent / Path(path).stem
        cache_path = cache_path.with_suffix(".cache")
        if cache_path.is_file():
            cache = torch.load(cache_path)  # load
            if (
                cache["hash"] != get_hash(self.label_files + self.img_files)
                or "results" not in cache
            ):  # changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Display cache
        [nf, nm, ne, nc, n] = cache.pop(
            "results"
        )  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=desc, total=n, initial=n)
        assert (
            nf > 0 or not augment
        ), f"No labels found in {cache_path}. Can not train without labels"

        # Read cache
        cache.pop("hash")  # remove hash
        labels, shapes = zip(*cache.values())
        temp = []
        for _l, s in zip(labels, shapes):
            _t = np.zeros_like(_l)
            _t[:, 1] = (_l[:, 1] + _l[:, 3]) / (2 * s[0])  # cx
            _t[:, 2] = (_l[:, 0] + _l[:, 2]) / (2 * s[1])  # cy
            _t[:, 3] = (_l[:, 3] - _l[:, 1]) / s[0]  # w
            _t[:, 4] = (_l[:, 2] - _l[:, 0]) / s[1]  # h
            _t[:, 0] = _l[:, 4]
            temp.append(_t)

        self.labels = temp
        self.shapes = np.array(shapes, dtype=np.float64)
        self.new_shapes = (
            np.full(self.shapes.shape, self.img_size) if self.square else None
        )
        self.img_files = list(cache.keys())  # update
        self.label_files = [lf[:-4] + ".json" for lf in self.img_files]  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]

            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = (
                np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int)
                * stride
            )
        self.imgs = [None] * n

    def cache_labels(self, path=Path("./labels.cache")):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(
            zip(self.img_files, self.label_files),
            desc="Scanning images",
            total=len(self.img_files),
        )
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = self.exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), "image size <10 pixels"

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, "r") as f:
                        l = json.load(f).get("bboxes")
                    l = np.array(l)
                    if len(l):
                        assert l.shape[1] == 4, "labels require 4 columns each"
                        assert (l >= -1).all(), "negative labels"
                        assert (
                            np.unique(l, axis=0).shape[0] == l.shape[0]
                        ), "duplicate labels"
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 4), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 4), dtype=np.float32)
                l = np.concatenate([l, np.zeros((len(l), 1), dtype="float32")], 1)
                x[im_file] = [l, shape]
            except Exception as e:
                nc += 1
                # print('WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

            pbar.desc = (
                f"Scanning '{path.parent / path.stem}' for images and labels... "
                f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            )

        if nf == 0:
            print(f"WARNING: No labels found in {path}")

        x["hash"] = get_hash(self.label_files + self.img_files)
        x["results"] = [nf, nm, ne, nc, i + 1]  # found, miss, empty, corrupted
        torch.save(x, path)  # save for next time
        logging.info(f"New cache created: {path}")
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic_face(index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp["mixup"]:
                img2, labels2 = self.load_mosaic_face(random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        elif self.square:
            img = self.imgs[index]
            if img is None:  # not cached
                path = self.img_files[index]
                img = cv2.imread(path)  # BGR
                assert img is not None, "Image Not Found " + path
                h0, w0 = img.shape[:2]  # orig hw
                rs = self.img_size == h0 == w0
                if not rs:
                    r = self.img_size / (max(h0, w0))
                    interp = (
                        cv2.INTER_AREA
                        if r < 1 and not self.augment
                        else cv2.INTER_LINEAR
                    )
                    img = cv2.resize(
                        img, (self.img_size, self.img_size), interpolation=interp
                    )
                labels = []
                x = self.labels[index]
                if x.size > 0:
                    # Normalized xywh to pixel xyxy format
                    labels = x.copy()
                    labels[:, 1] = self.img_size * (x[:, 1] - x[:, 3] / 2)
                    labels[:, 2] = self.img_size * (x[:, 2] - x[:, 4] / 2)
                    labels[:, 3] = self.img_size * (x[:, 1] + x[:, 3] / 2)
                    labels[:, 4] = self.img_size * (x[:, 2] + x[:, 4] / 2)
                shapes = (h0, w0), (
                    (self.img_size / h0, self.img_size / h0),
                    (0.0, 0.0),
                )

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = (
                self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            )  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = (
                    ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
                )  # pad width
                labels[:, 2] = (
                    ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
                )  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

            # Augment colorspace
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def exif_size(img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        try:
            rotation = dict(img._getexif().items())[orientation]
            if rotation == 6:  # rotation 270
                s = (s[1], s[0])
            elif rotation == 8:  # rotation 90
                s = (s[1], s[0])
        except:
            pass
        return s

    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.imgs[index]
        if img is None:  # not cached
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, "Image Not Found " + path
            h0, w0 = img.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if (
                r != 1
            ):  # always resize down, only resize up if training with augmentation
                interp = (
                    cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                )
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        else:
            return (
                self.imgs[index],
                self.img_hw0[index],
                self.img_hw[index],
            )  # img, hw_original, hw_resized

    def load_mosaic_face(self, index):
        # loads images in a mosaic
        labels4 = []
        s = self.img_size
        yc, xc = [
            int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border
        ]  # mosaic center x, y
        indices = [index] + [
            self.indices[random.randint(0, self.n - 1)] for _ in range(3)
        ]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full(
                    (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
                )  # base image with 4 tiles
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = self.labels[index]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                # box, x1,y1,x2,y2
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh

            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(
                labels4[:, 1:5], 0, 2 * s, out=labels4[:, 1:5]
            )  # use with random_perspective
            # img4, labels4 = replicate(img4, labels4)  # replicate

        img4, labels4 = random_perspective(
            img4,
            labels4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove
        return img4, labels4


class FaceDataLoader(BaseDataLoader):
    def __init__(
        self,
        path,
        imgsz,
        batch_size,
        stride,
        ratio=None,
        hyp=None,
        augment=False,
        pad=0.0,
        rect=False,
        square=False,
        workers=8,
        collate_fn=LoadFaceImagesAndLabels.collate_fn,
    ):
        assert not (square and rect), "Bad args!"
        self.batch_size = batch_size
        self.ratio = ratio
        self.args = {
            "imgsz": imgsz,
            "stride": stride,
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "workers": workers,
        }
        if isinstance(path, str):
            self.dataset = LoadFaceImagesAndLabels(
                path,
                imgsz,
                batch_size,
                augment=augment,
                hyp=hyp,
                rect=rect,
                square=square,
                stride=int(stride),
                pad=pad,
            )
        else:
            self.dataset = path
        if ratio:
            self.train_set, self.test_set = random_split(
                self.dataset,
                [
                    int(len(self.dataset) * (1 - ratio)),
                    len(self.dataset) - int(len(self.dataset) * (1 - ratio)),
                ],
            )
            self.train_set = self.train_set.dataset
            self.test_set = self.test_set.dataset
        super(FaceDataLoader, self).__init__(
            self.dataset, batch_size, collate_fn=collate_fn, workers=workers
        )

    def get_loader(self):
        if self.ratio:
            return FaceDataLoader(self.train_set, **self.args), FaceDataLoader(
                self.test_set, **self.args
            )
        else:
            return self
