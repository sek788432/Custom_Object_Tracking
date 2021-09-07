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
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .deep_sort import generate_detections as gdet

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def draw_boxes(image, boxes, category_index):
    colors = list(ImageColor.colormap.values())
    for xmin, ymin, xmax, ymax, tracking_id, class_name in boxes:
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        label_txt = class_name + ' ' + str(tracking_id)
        color = hex_to_rgb(colors[hash(class_name) % len(colors)])
        thickness = 2
        fontScale = 0.5
        fontFace = cv2.FONT_HERSHEY_COMPLEX
        labelSize = cv2.getTextSize(label_txt, fontFace, fontScale, thickness)
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, thickness)
        cv2.rectangle(image, (xmin,ymin-labelSize[0][1]), (xmin+labelSize[0][0],ymin), color, -1)
        cv2.putText(image, label_txt, (xmin,ymin), fontFace, fontScale, (0,0,0), thickness)   
        
def TrackVideo(PATH_TO_LABELS, PATH_TO_SAVED_MODEL, PATH_TO_TEST_VIDEO, 
               PATH_TO_OUTPUT_VIDEO, MIN_SCORE_THRESH, DEEP_SORT_MODEL):
    
    PATH_TO_SAVED_MODEL = os.path.join(PATH_TO_SAVED_MODEL, "saved_model")
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                            use_display_name=True)

    # Load saved model and build the detection function
    print('Loading model...', end='')
    start_time = time.time()
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None

    # initialize deep sort object

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Video setting
    cap = cv2.VideoCapture(PATH_TO_TEST_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(PATH_TO_OUTPUT_VIDEO, codec, fps, (width, height)) # output_path must be .mp4


    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None

    #initialize deep sort object
    model_filename = DEEP_SORT_MODEL
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

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

        # filter score > MIN_SCORE_THRESH
        indexes = [ i for i in range(len(detections["detection_scores"])) 
              if detections["detection_scores"][i] > MIN_SCORE_THRESH]

        detections["detection_boxes"] = detections["detection_boxes"][indexes, ...]
        detections["detection_scores"] = detections["detection_scores"][indexes, ...]
        detections["detection_classes"] = detections["detection_classes"][indexes, ...]

        height, width = image_np_with_detections.shape[:2]
        boxes, scores, names = [], [], []
        Track_only = [category_index[ID]['name'] for ID in category_index]

        for i in range(len(indexes)):
            ymin, xmin, ymax, xmax = detections["detection_boxes"][i]
            score, classes = detections["detection_scores"][i], detections["detection_classes"][i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            if len(Track_only) !=0 and category_index[classes]['name'] in Track_only or len(Track_only) == 0:
                boxes.append([xmin, ymin, xmax-xmin, ymax-ymin])
                scores.append(score)
                names.append(category_index[classes]['name'])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(image_np.copy(), boxes))
        track_detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(track_detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            # index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            # Structure data, that we could use it with our draw_bbox function
            tracked_bboxes.append(bbox.tolist() + [tracking_id, class_name])  
        
        draw_boxes(
          image_np_with_detections,
          tracked_bboxes,
          category_index
        )

        out.write(image_np_with_detections[:,:,::-1])

    print('Done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('--saved_model_path', required=True,
                        help='path to saved model')
    parser.add_argument('--test_path', required=True,
                        help='path to test video')
    parser.add_argument('--label_map_path', required=True, help='path to label map')
    parser.add_argument('--deep_sort_model', required=True, help='path to deep sort model')
    parser.add_argument('--output_path', required=True,
                        help='path to output predicted video')
    parser.add_argument('--min_score_thresh', required=False, default=0.0,
                        help='min score threshold')

    args = parser.parse_args()

    # Path definition
    PATH_TO_SAVED_MODEL = args.saved_model_path
    PATH_TO_TEST_VIDEO = args.test_path
    PATH_TO_OUTPUT_VIDEO = args.output_path 
    PATH_TO_LABELS = args.label_map_path
    DEEP_SORT_MODEL = args.deep_sort_model
    MIN_SCORE_THRESH = float(args.min_score_thresh)

    TrackVideo(PATH_TO_LABELS, PATH_TO_SAVED_MODEL, PATH_TO_TEST_VIDEO, PATH_TO_OUTPUT_VIDEO,  MIN_SCORE_THRESH, DEEP_SORT_MODEL)
