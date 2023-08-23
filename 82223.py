# import packages
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Set web app title
st.title('Oddy Test Coupon Rating')

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
# import pathlib
import tensorflow as tf

# tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Currently this reloads the model from the base and checkpoint, and does not use the saved_model version.
# Download and extract EfficientDet model

@st.cache_resource
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    print("Model reloaded")
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'efficientdet_d1_coco17_tpu-32'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

# Download labels file
@st.cache_resource
def download_labels(filename):
    base_url = 'https://github.com/fredthefish/OddyTest/blob/main/Data/StringIntLabelMap.pbtxt'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    # label_dir = pathlib.Path(label_dir)
    print("Labels reloaded")
    return str(label_dir)

LABEL_FILENAME = 'StringIntLabelMap.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)

# Load the model
# import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

# start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-49')).expect_partial()

#Upload images
uploaded_files = st.file_uploader("Choose an image file", type=['png', 'jpg', 'tiff'], accept_multiple_files=True)

# end_time = time.time()
# elapsed_time = end_time - start_time
# st.success('Model load complete! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
# Run model
def load_image_into_numpy_array(image):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(image)

@tf.function
def detect(image_tensor):
    """Run the neural network on an input image.

    Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

    Returns:
    3 Tensors (`detection_boxes`, `detection_classes`, and `detection_scores`).
    """
    preprocessed_image, shapes = detection_model.preprocess(image_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    # use the detection model's postprocess() method to get the the final detections
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

def detect_extract(image_tensor):
    ''' Function to extract the predicted classes, scores, and bounding boxes'''

    # Run neural network on image
    detections = detect(image_tensor)

    # Get predicted classes, scores, and boxes
    class_array = detections['detection_classes'][0].numpy().astype('int') + 1
    det_scores = detections['detection_scores'][0].numpy()
    det_boxes = detections['detection_boxes'][0].numpy()

    return class_array, det_scores, det_boxes

def intersection_over_union(pred_box, true_box):
    # Split the predicted box array into separate values: (ymin, xmin, ymax, xmax)
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = np.split(pred_box, 4)
    ymin_true, xmin_true, ymax_true, xmax_true = np.split(true_box, 4)

    smoothing_factor = 1e-10

    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    overlap_area = np.maximum((xmax_overlap - xmin_overlap), 0)  * np.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)

    return iou

def chooseBoxes(image_tensor):
  ''' This function finds the highest scoring boxes that do not overlap with eachother for plotting
  Args: image_tensors - one image in tensor format
  Returns: reduced lists of predicted classes, scores, and bounding boxes
  '''

  # Run the image through the neural network to get predictions
  det_classes, det_scores, det_boxes = detect_extract(image_tensor)

  # Create a list of boxes to keep for plotting
  keep_list = [0]
  # Iterate over all the predicted boxes
  for i in range(len(det_scores)):
      # Ignore any boxes with scores of 0
      if det_scores[i] == 0:
        break
      # The first box (i = 0) will always be included because it has the highest score
      if i == 0 :
        continue
      # Create list to store the IOUs of box i
      ious = []
      # Need to find intersections with all the boxes currently being 'kept' for display on images
      for j in keep_list:
        iou = intersection_over_union(det_boxes[i],det_boxes[j])
        # print('Box', str(i),'has iou with box',str(j),': ',iou)
        ious.append(iou)
      # Sorts the IOUs to find the highest IOU between box i and the boxes in the keep list
      ious.sort(reverse = True)
      # If the largest IOU is less than 0.2 (poor fit with all other boxes), add this box to the list
      if ious[0] < 0.2:
          keep_list.append(i)
          # print('Added box', str(i),'to keep list')

  # Return only the necessary boxes
  return det_classes[keep_list], det_scores[keep_list], det_boxes[keep_list]

def plot_detections(image_np, boxes, classes, scores, category_index, figsize=(12, 16), image_name=None):

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=6,
        min_score_thresh=0.3,
        line_thickness = 10)
    print("Finished inference")
    st.sidebar.image(image_np_with_detections)
    return

def findMetals(pred_boxes, class_array, uploaded_file):
  '''
  Args:
  pred_boxes: dataframe for the predictions
  class_array: array of classes for bounding boxes
  uploaded_file: file name

  Returns:
  Updated pred_boxes dataframe with the classes from 1-9 or 0 for NA values
  '''

  # Add filename to dataframe
  predBoxes.at[i,'File Name']  = uploaded_file.name

  # Finding silver predictions and adding to dataframe
  Ag = class_array[class_array <= 3]
  if len(Ag) == 0:
    # If no silver predicted, leave as 0's and give warning message
    print('No silver coupons detected in file',uploaded_file.name)
  elif len(Ag) == 1:
    # If only one silver, add to first column
    predBoxes.iloc[i,1] = Ag[0]
  else:
    # If two or more silver, just grab the first two
    predBoxes.iloc[i,1:3] = Ag[0:2]
  # Finding copper predictions and adding to dataframe
  Cu = class_array[(class_array > 3) & (class_array <= 6)]
  if len(Cu) == 0:
    print('No copper coupons detected in file',uploaded_file.name)
  elif len(Cu) == 1:
    predBoxes.iloc[i,3]  = Cu[0]
  else:
    predBoxes.iloc[i,3:5] = Cu[0:2]
  # Finding lead predictions and adding to dataframe
  Pb = class_array[class_array >= 7]
  if len(Pb) == 0:
    print('No lead coupons detected in file',uploaded_file.name)
  elif len(Pb) == 1:
    predBoxes.iloc[i,5]  = Pb[0]
  else:
    predBoxes.iloc[i,5:7] = Pb[0:2]

  return predBoxes

def scanImage(image_np, image_tensors):
  ''' Runs an image through the CNN, finds only the best bounding boxes (chooseBoxes function),
  and plots and saves the image labelled with predictions.
  Args:
  image_nps - numpy image stack.
  image_tensors - tensor image stack
  files - file names of images for saving.
  i - number of image
  Returns: the predicted classes, bounding boxes, and scores (class_array, det_boxes, det_scores)
  '''

  # Get the predicted bounding boxes from the neural network that don't overlap
  class_array, det_scores, det_boxes = chooseBoxes(image_tensors)

  # Use the plot_detections function to draw the ground truth boxes
  plot_detections(image_np, det_boxes,class_array,det_scores,category_index = category_index)

  return class_array, det_boxes, det_scores

# Calc overall material rating
def condition(rate):
    if '-U' in str(rate):
        return "Unsuitable"
    elif '-T' in str(rate):
        return "Temporary"
    elif '-P' in str(rate):
        return 'Pass'
    return ''

# Determine cell color
def color(rate):
    if '-U' in str(rate):
        return 'background-color: red'
    elif '-T' in str(rate):
        return 'background-color: orange'
    elif '-P' in str(rate):
        return 'background-color: green'
    if 'Unsuitable' in str(rate):
        return 'background-color: red'
    elif 'Temporary' in str(rate):
        return 'background-color: orange'
    elif 'Pass' in str(rate):
        return 'background-color: green'
    return ''

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

# Creating empty data frame to store the classes outputted by the model for csv file
count = len(uploaded_files)
z = [0] * count
blank_row = {'File Name': ['']*count, 'Ag1': z, 'Ag2': z, 'Cu1': z, 'Cu2': z, 'Pb1': z, 'Pb2': z}
predBoxes = pd.DataFrame(blank_row)

i = 0
if uploaded_files is not None:
    st.sidebar.title('Image Preview')
    for i in range(len(uploaded_files)):
        # Get file info
        uploaded_file = uploaded_files[i]
        st.sidebar.write(uploaded_file.name)
        print(uploaded_file.name)

        # Convert to pil format
        image = Image.open(uploaded_file)
        print('Running inference for {}... '.format(image), end='')

        # Get np array from image
        image_np = load_image_into_numpy_array(image)

        # Convert to tensor
        image_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        # Choose best bounding boxes
        class_array, det_boxes, det_scores = scanImage(image_np, image_tensor)

        # Run the findMetals function to get the class data formatted as a dataframe
        predBoxes = findMetals(predBoxes, class_array, uploaded_file)

        # Display image
        print("Finished inference")

    # Set up dataframe output
    # Set up dictionary
    reverse_dict = {0: 'N/A', 1: 'Ag-P', 2: 'Ag-T', 3: 'Ag-U', 4: 'Cu-P', 5: 'Cu-T', 6: 'Cu-U', 7: 'Pb-P', 8: 'Pb-T',
                    9: 'Pb-U'}

    # For labelled coupons change to corrosion type value
    for k in [1, 2, 3, 4, 5, 6]:
        # Map the label dictionary to a column to populate the corresponding class integer values
        predBoxes.iloc[:, k] = predBoxes.iloc[:, k].map(reverse_dict)

    predBoxes['Duplicates Match?'] = predBoxes.apply(lambda x: "Match" if x['Cu1'] ==
                                                               x['Cu2'] and x['Ag1']
                                                               == x['Ag2'] and x['Pb1']
                                                               == x['Pb2'] else "Mismatch", axis=1)

    # Write overall rating by applying the conditions
    predBoxes['Overall Rating'] = predBoxes.apply(condition, axis=1)
    # Apply color based on coupon ratings
    styleddf = predBoxes.style.applymap(color)
    # Display dataframe of results
    st.write("Results")
    st.dataframe(data=styleddf, hide_index=True)

    csv = convert_df(predBoxes)

    st.download_button(
    "Download csv",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )
# Save the model
# export_path = PATH_TO_MODEL_DIR
# tf.saved_model.save(detection_model, export_path)

# st.sidebar.button('Rate coupons')
# Currently only downloads last image
# st.sidebar.download_button("Download processed images", data=image_jpg, file_name="Processed images.jpg")
