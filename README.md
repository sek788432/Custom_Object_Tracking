# Custom-Object-Tracking

## Installation
Please install [Tensorflow 2 Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) and add the path to your environment.

## Usage
1. Clone the github repository
```
git clone https://github.com/sek788432/Custom-Object-Tracking.git
cd  Custom-Object-Tracking/objectTracker/
```

2. Run the tracking.py
* saved_model_path: your own object detection model ckpt path
* test_path: test video path
* label_map_path: label_map.pbtxt path
* deep_sort_model: deep sort model path
* output_path: output video path
* min_score_thresh: the minimum score threshold of object detection model
* e.g.
    ```   
    python tracking.py \
        --saved_model_path=exported-models/ssd_resnet50_119ckpt \
        --test_path=test_video.mp4 \
        --label_map_path=label_map.pbtxt \
        --deep_sort_model=data/mars-small128.pb \
        --output_path=test_video_tracking.mp4 \
        --min_score_thresh=.5
    ```

## Result
### Tracking vehicle by our own model
![Vehicle Tracking](test_video_tracking.mp4?raw=true "video")
