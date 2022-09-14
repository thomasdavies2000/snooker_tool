import tensorflow as tf
import time
import numpy as np
import warnings
import cv2
warnings.filterwarnings('ignore')
from PIL import Image
from matplotlib.patches import Circle

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from helpers.Positions import Positions, Ball
IMAGE_SIZE = (12, 8) # Output display size as you want
import matplotlib
matplotlib.use('TkAgg')
PATH_TO_SAVED_MODEL="saved_model"
print('Loading model...', end='')
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

#Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap("labelmap/label_map.pbtxt",use_display_name=True)

def load_image_into_numpy_array(path):

    return np.array(Image.open(path))

image_path = "images_to_detect/chin.jpg"


image_np = load_image_into_numpy_array(image_path)

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
positions = Positions()
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      positions,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh_ball=.2,
      min_score_thresh_pocket=0.2,
      # Adjust this value to set the minimum probability boxes to be classified as True
      agnostic_mode=False)


img = matplotlib.image.imread('table/table.jpg')

# Create a figure. Equal aspect so circles look circular
fig,ax = matplotlib.pyplot.subplots(1)
ax.set_aspect('equal')

imgplot = matplotlib.pyplot.imshow(img)
pocket_coords = [(40,60), (40, 1167), (2230, 60), (2230, 1167)]

mtx = positions.initialTransform(pocket_coords)

balls, colors = positions.separateBallsAndColors()

original = np.array([tuple(balls)], dtype=np.float32)
converted = cv2.perspectiveTransform(original, mtx)
for ball in range(len(converted[0])):
      circ = Circle(converted[0][ball],15, color=colors[ball])
      ax.add_patch(circ)
matplotlib.pyplot.show()
matplotlib.pyplot.figure(figsize=IMAGE_SIZE, dpi=200)
matplotlib.pyplot.axis("off")
matplotlib.pyplot.imshow(image_np_with_detections)
matplotlib.pyplot.show()

