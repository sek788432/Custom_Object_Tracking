import argparse
import os
import cv2
import numpy as np
from PIL import Image
from PIL import ImageColor
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def draw_bounding_boxes_on_image(image,
                 box,
                 color,
                 thickness=2,
                 label_txt=''):
    height, width = image.shape[:2]
    ymin, xmin, ymax, xmax = box
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)
    fontScale = 0.5
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    labelSize = cv2.getTextSize(label_txt, fontFace, fontScale, thickness)
    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, thickness)
    cv2.rectangle(image, (xmin,ymin-labelSize[0][1]), (xmin+labelSize[0][0],ymin), color, -1)
    cv2.putText(image, label_txt, (xmin,ymin), fontFace, fontScale, (0,0,0), thickness)    

def draw_boxes(image, boxes, category_index, indexes):
    colors = list(ImageColor.colormap.values())
    for i in range(len(indexes)):
        # ymin, xmin, ymax, xmax = boxes["detection_boxes"][i]
        scores, classes = boxes["detection_scores"][i], boxes["detection_classes"][i]
        color = colors[hash(category_index[classes]['name']) % len(colors)]
        # print(color)
        label_txt = '{}: {}%'.format(category_index[classes]['name'], int(scores*100))
        draw_bounding_boxes_on_image(image,
                      boxes["detection_boxes"][i],
                      hex_to_rgb(color),
                      label_txt=label_txt)

parser = argparse.ArgumentParser(description='Download and process tf files')
parser.add_argument('--saved_model_path', required=True,
                    help='path to saved model')
parser.add_argument('--test_path', required=True,
                    help='path to test video')

parser.add_argument('--output_path', required=True,
                    help='path to output predicted video')
parser.add_argument('--min_score_thresh', required=False, default=0.0,
                    help='min score threshold')
args = parser.parse_args()

# Path definition
PATH_TO_SAVED_MODEL = os.path.join(args.saved_model_path, "saved_model")
PATH_TO_TEST_VIDEO = args.test_path
PATH_TO_OUTPUT_VIDEO = args.output_path 
MIN_SCORE_THRESH = float(args.min_score_thresh)

# Load the Labels
PATH_TO_LABELS = "label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                        use_display_name=True)

# Load saved model and build the detection function
print('Loading model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Video setting
cap = cv2.VideoCapture(PATH_TO_TEST_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(PATH_TO_OUTPUT_VIDEO, codec, fps, (width, height)) # output_path must be .mp4

while True:
    ret, frame = cap.read()
    if ret == False:
      break
    
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #       image_np_with_detections,
    #       detections['detection_boxes'],
    #       detections['detection_classes'],
    #       detections['detection_scores'],
    #       category_index,
    #       use_normalized_coordinates=True,
    #       max_boxes_to_draw=200,
    #       min_score_thresh=MIN_SCORE_THRESH,
    #       agnostic_mode=False)

    # filter score > MIN_SCORE_THRESH
    indexes = [ i for i in range(len(detections["detection_scores"])) 
          if detections["detection_scores"][i] > MIN_SCORE_THRESH]
    
    detections["detection_boxes"] = detections["detection_boxes"][indexes, ...]
    detections["detection_scores"] = detections["detection_scores"][indexes, ...]
    detections["detection_classes"] = detections["detection_classes"][indexes, ...]
    
    draw_boxes(
      image_np_with_detections,
      detections,
      category_index,
      indexes)
    
    out.write(image_np_with_detections[:,:,::-1])
    
print('Done')