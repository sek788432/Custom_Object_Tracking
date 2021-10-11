# Custom Object Tracking
## Introduction
This repo provides function call to track multi-objects in videos with a given trained object detection model and a source video file as inputs. The tracking approach used in the repo is [DeepSort](https://github.com/nwojke/deep_sort) - [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/pdf/1703.07402.pdf)

## Installation
Please install [Tensorflow 2 Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) and add the path to your environment.

## Usage
### 1. Clone the github repository
```shell
git clone https://github.com/sek788432/Custom_Object_Tracking.git
```

### 2. Execution (In two ways)
* saved_model_path: your own object detection model ckpt path
* test_path: test video path
* label_map_path: label_map.pbtxt path
* deep_sort_model: deep sort model path
* output_path: output video path
* min_score_thresh: the minimum score threshold of object detection model

1. Run tracking.py
* e.g.
    ```shell
    cd  Custom_Object_Tracking/objectTracker/
    python tracking.py \
        --saved_model_path=exported-models/ssd_resnet50_119ckpt \
        --test_path=test_video.mp4 \
        --label_map_path=label_map.pbtxt \
        --deep_sort_model=data/mars-small128.pb \
        --output_path=test_video_tracking.mp4 \
        --min_score_thresh=.5
    ```
2. Call TrackVideo function
* e.g.
    ```python
    from Custom_Object_Tracking.objectTracker.tracking import TrackVideo
    TrackVideo(label_path, model_path, video_path,
                   output_path, threshold, deep_sort_model)
    ```

## Result
### Tracking vehicle by our own model (SSD ResNet50 trained on Waymo Dataset)
![Vehicle Tracking](test_video_tracking.gif?raw=true "video")
