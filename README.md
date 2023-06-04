# Face detection: Implementing YOLOv5-face and YOLOv5-blazeface using Pytorch

## BlazeFace overview

BlazeFace is a fast, light-weight face detector from Google Research. [Read more](https://sites.google.com/view/perception-cv4arvr/blazeface), [Paper on arXiv](https://arxiv.org/abs/1907.05047)

A pretrained model is available as part of Google's [MediaPipe](https://github.com/google/mediapipe/blob/master/mediapipe/docs/face_detection_mobile_gpu.md) framework.

Besides a bounding box, BlazeFace also predicts 6 keypoints for face landmarks (2x eyes, 2x ears, nose, mouth).

The BlazePaper paper mentions that there are two versions of the model, one for the front-facing camera and one for the back-facing camera. This repo includes only the frontal camera model, as that is the only one I was able to find an official trained version for. The difference between the two models is the dataset they were trained on. As the paper says,

> For the frontal camera model, only faces that occupy more than 20% of the image area were considered due to the intended use case (the threshold for the rear-facing camera model was 5%).

This means the included model will not be able to detect faces that are relatively small. It's really intended for selfies, not for general-purpose face detection.


## In this project

> This was one of my personal projects about object detection, specifically face detection.

**Purpose of this project**
- Understanding blazeface architecture.
- Understanding full pipeline of face detection task.
- Learn to convert pipeline to others template.
- Learn to code and control basic components in a pipeline such as: dataset, dataloader, trainer, validation step, predict, pre/post-processing, etc.

### Dataset

The dataset that I used was WiderFace dataset containing about 42648 samples
-  Training and validation set was splitted as ratio 80/20.

### Model
Here I implemented YOLOv5-face based on [Pytorch template](https://github.com/victoresque/pytorch-template) and tried to implement [BlazeFace](https://sites.google.com/view/perception-cv4arvr/blazeface) architecture as a backbone of YOLOv5. The reference github that I used you can find it [here](https://github.com/deepcam-cn/yolov5-face/tree/master).

### Experiment and result

The model was trained for 400 epochs (as a recommended epochs in yolov5 document) with specific config you can find in this repo ***config/hyp.trainer_blazeface.yaml***.

Due to computational hardware limitations, I could only conduct to train YOLOv5-blazeface. The model was trained on one GPU RTX 3090 with 24GB RAM. 

After 400 epochs, here is the result I got:
| Model              | P       | R       | mAP:.5   | mAP@.5-.95 |
|--------------------|---------|---------|---------|-------------|
| YOLOv5-blazeface   | 0.9431  | 0.9592  | 0.9714  | 0.8782      |
        
I believed this was fairly good enough result for a mini-project (and for a fully training from scratch model)! 

### Conclusion

YOLOv5-blazeface has almost the same performance as YOLOv5-face in case the face is fully displayed in the camera frame. If the face is not fully displayed or obscured by the object, then YOLOv5-blazeface's confidence score will be lower than that of YOLOv5-face, I think the reason is because yolov5-face has been trained on the big dataset as well as fully augmented.


## Usage
> For inference purpose only!

### Install with Miniconda
```bash
git clone https://github.com/doubleK29/blazeface.git
cd blazeface
conda create -n blazeface python=3.9
pip install -r requirements.txt
```
### Inference
`predict.py` runs inference on three sources: image, video and laptop camera


```bash
python predict.py --video 0  # camera
python predict.py --video path/to/video  # video
python predict.py --img path/to/image  # image
```

**Example of inference with my camera laptop**
![Video](https://drive.google.com/file/d/1jSGFXKDMbtnZQteR6MkMSuWZhA4OCiSz/view?usp=drive_link)

### Deployment

I have tried to deploy the project on Streamlit. But the limitation of time do not allow me to fix issues to allow website using user's webcam for testing the detection model. Hope this will not be a strictly minus point to my overall score...

## References

- https://github.com/deepcam-cn/yolov5-face
- https://github.com/victoresque/pytorch-template
- https://sites.google.com/view/perception-cv4arvr/blazeface
