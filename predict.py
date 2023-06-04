# -*- coding: UTF-8 -*-
import argparse
import copy
import time

import cv2
import torch
import torch.nn as nn

import warnings
from data_loader.augmentation import letterbox
from utils.general import (
    check_img_size,
    non_max_suppression_face,
    scale_coords,
    xyxy2xywh,
)
from models.model import YOLOV5n_05, YOLOv5_BlazeFace
import sys

sys.path.insert(0, "models")


def show_results(img, xywh, conf, model_name, class_num, color=(0, 255, 0)):
    h, w, _ = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    text = model_name + label
    cv2.putText(
        img,
        text,
        (x1, y1 - 2),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


def detect_one(model, model_yolo, video_path, device):
    model_names = ["blaze: ", "yolov5: "]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(
        f"{video_path}2_square_result.mp4",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (frame_width, frame_height),
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        prev = time.time()
        _, orgimg = cap.read()
        orgimg = cv2.flip(orgimg, 1)
        img_size = 320
        conf_thres = 0.3
        iou_thres = 0.5

        img0 = copy.deepcopy(orgimg)

        h0, w0 = orgimg.shape[:2]
        r = img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        # img0 = cv2.resize(img0, (img_size, img_size))
        imgsz = check_img_size(img_size, s=model.stride.max())

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()

        # Run inference
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        new = time.time()
        # Inference
        with torch.no_grad():
            pred = model(img)[0]
            pred1 = model_yolo(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        pred1 = non_max_suppression_face(pred1, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(
                device
            )  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], orgimg.shape
                ).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].detach().cpu().numpy()
                    class_num = det[j, 5].detach().cpu().numpy()
                    orgimg = show_results(orgimg, xywh, conf, model_names[0], class_num)

        for i, det in enumerate(pred1):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(
                device
            )  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], orgimg.shape
                ).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].detach().cpu().numpy()
                    class_num = det[j, 5].detach().cpu().numpy()
                    orgimg = show_results(
                        orgimg, xywh, conf, model_names[1], class_num, color=(255, 0, 0)
                    )
        new = time.time()
        fps = 1 / (new - prev)
        cv2.putText(
            orgimg, str(int(fps)), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA
        )
        out.write(orgimg)
        cv2.imshow("FaceDetector", orgimg)
        if cv2.waitKey(1) % 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def detect_image(model, model_yolo, device, image_path):
    font = cv2.FONT_HERSHEY_SIMPLEX
    model_names = ["blaze: ", "yolov5: "]

    # Load image
    orgimg = cv2.imread(image_path)
    orgimg = cv2.flip(orgimg, 1)
    img_size = 320
    conf_thres = 0.3
    iou_thres = 0.5

    img0 = copy.deepcopy(orgimg)

    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())

    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        pred1 = model_yolo(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    pred1 = non_max_suppression_face(pred1, conf_thres, iou_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(
            device
        )  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].detach().cpu().numpy()
                class_num = det[j, 5].detach().cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, model_names[0], class_num)

    for i, det in enumerate(pred1):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(
            device
        )  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].detach().cpu().numpy()
                class_num = det[j, 5].detach().cpu().numpy()
                orgimg = show_results(
                    orgimg, xywh, conf, model_names[1], class_num, color=(255, 0, 0)
                )

    # Save output image
    output_path = "output.jpg"
    # cv2.putText(orgimg, str(int(fps)), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imwrite(output_path, orgimg)
    print(f"Output image saved to: {output_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="weights/Blazeface_yolo-best.pt",
        help="models.pt path(s)",
    )
    parser.add_argument("--video", type=str, default=None, help="path to video")
    parser.add_argument("--img", type=str, default=None, help="path to image")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default="config/models.yolov5n-0.5.yaml",
        help="path to yaml models file",
    )
    opt = parser.parse_args()

    # Use camera
    if opt.video == str(0):
        opt.video = 0
    else:
        opt.video = opt.video

    # Load YOLOv5 with backbone BlazeFace
    device = torch.device("cpu")
    model = YOLOv5_BlazeFace().to(device)
    ckpt = torch.load(opt.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.fuse().eval()

    # Load YOLOv5
    modelyolo = YOLOV5n_05().to(device)
    modelyolo.load_state_dict(
        torch.load("weights/YOLO_exp-best.pt", map_location=device)["model"]
    )
    modelyolo.fuse().eval()

    # Detect
    if opt.video is not None and opt.img is not None:
        print("Please provide either --video or --img argument, but not both.")
    elif opt.video is not None:
        # Process video
        detect_one(model, modelyolo, opt.video, device)
    elif opt.img is not None:
        # Process image
        detect_image(model, modelyolo, device, opt.img)
    else:
        print("Please provide either --video or --img argument.")
