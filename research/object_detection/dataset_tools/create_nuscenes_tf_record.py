# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw nuScenes dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_nuscenes_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os

# 3rd Party Libraries
import tensorflow as tf
from PIL import Image
import numpy as np
import tensorflow as tf
import contextlib2

# Local Libraries
from nuscenes_utils.nuscenes import NuScenes
from nuscenes_utils.data_classes import PointCloud, Box
from nuscenes_utils.geometry_utils import box_in_image, view_points, BoxVisibility

# from tensorflow_models.research.object_detection.utils import dataset_util

import tf_record_creation_util
from tensorflow_models.research.object_detection.utils import dataset_util

# Global Variables
IMG_CHANNELS = [0, 1, 2]
RADAR_CHANNELS = [3, 7, 10, 11]
CHANNELS = [*IMG_CHANNELS, *RADAR_CHANNELS]
IMAGE_SIZE = (416, 416)
NUSCENES_PATH = '~/data/nuscenes'
CAM_CHANNEL = 'CAM_FRONT'
RADAR_CHANNEL = 'RADAR_FRONT'
SENSOR_CHANNELS = ['CAM_FRONT']#, 'RADAR_FRONT']
DATATYPE = np.float32

flags = tf.app.flags
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def get_category_index_mapping(nusc):
    """
    :param nusc: [Nuscenes] The nuscenes dataset object
    :returns: [dict of (str, int)] mapping from category name to the corresponding index-number
    """
    category_indices = {c['name']: i for i, c in enumerate(nusc.category)}
    return category_indices


def get_label_data_2D_bounding_boxes(nusc, sample_token, sensor_channels):
    """
    Create 2D bounding box labels from the given sample token.

    1 bounding box vector contains:
    - [0]: box dimensions
        - box x_min (normalized to the image size)
        - box y_min (normalized to the image size)
        - box x_max (normalized to the image size)
        - box y_max (normalized to the image size)
    - [1]: class_category_name
    - [2]: class_index

    :param sample: the sample to get the annotation for
    :param sensor_channels: list of channels for cropping the labels, e.g. ['CAM_FRONT', 'RADAR_FRONT']
        This works only for CAMERA atm

    :returns: [(nx4 np.array, list of str, list of int)] Labels (boxes, class_names, class_indices) (used for training)
    """

    assert not any([s for s in sensor_channels if 'RADAR' in s]), "RADAR is not supported atm"

    sample = nusc.get('sample', sample_token)
    box_labels = [] # initialize counter for each category
    class_names = []
    class_indices = []
    category_indices = get_category_index_mapping(nusc)

    # Camera parameters
    for selected_sensor_channel in sensor_channels:

        sd_rec = nusc.get('sample_data', sample['data'][selected_sensor_channel])
        # sensor = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

        # Create Boxes:
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(sd_rec['token'], box_vis_level=BoxVisibility.ANY)
        imsize = (sd_rec['width'], sd_rec['height'])
        
        # Add labels to boxes into 
        # assign_box_labels(nusc, boxes)
        
        # Create labels for all boxes that are visible
        for box in boxes:

            box.label = category_indices[box.name]   

            if box_in_image(box=box, intrinsic=camera_intrinsic, imsize=imsize, vis_level=BoxVisibility.ANY):
                # Check if box is visible
                # If visible, we create the corresponding label
                box2d = box.box2d(camera_intrinsic, imsize=imsize, normalize=True)

                box_labels.append(box2d)
                class_names.append(box.name)
                class_indices.append(box.label)

    return box_labels, class_names, class_indices


def create_tf_example(nusc, sample_token, **kwargs):
    """Creates a tf.Example proto from sample image+.

    :param sample_token: [str] the sample token pointing to a specific nuscenes sample

    :returns: [tf.train.Example] The created tf.Example containing image plus
    """

    # Get the sensor_data from nuscenes data for a specific sample
    sample = nusc.get('sample', sample_token)

    # Get Radar Sensor
    radar_token = sample['data'][RADAR_CHANNEL]
    radar_sample_data = nusc.get('sample_data', radar_token)
    radar_calibrated_sensor = nusc.get('calibrated_sensor', radar_sample_data['calibrated_sensor_token'])
    radar_rotation = radar_calibrated_sensor['rotation']
    radar_translation = radar_calibrated_sensor['translation']
    # Get radar sensor data
    radar_file_path = os.path.join(nusc.dataroot, radar_sample_data['filename'])
    radar_data = PointCloud.load_pcd_bin(radar_file_path)  # Load radar points
    radar_data = np.array(radar_data, dtype=DATATYPE)  
    radar_data = np.reshape(radar_data, newshape=[radar_data.size,]) # tf_record only takes 1-D array
     

    # Get Camera Data
    camera_token = sample['data'][CAM_CHANNEL]
    camera_sample_data = nusc.get('sample_data', camera_token)
    camera_calibrated_sensor = nusc.get('calibrated_sensor', camera_sample_data['calibrated_sensor_token'])
    camera_intrinsic = np.array(camera_calibrated_sensor['camera_intrinsic'], dtype=DATATYPE)
    camera_intrinsic = np.reshape(camera_intrinsic, newshape=[camera_intrinsic.size,])# tf_record only takes 1-D array

    camera_rotation = camera_calibrated_sensor['rotation']
    camera_translation = camera_calibrated_sensor['translation']
    # Get camera sensor data
    camera_file_path = os.path.join(nusc.dataroot, camera_sample_data['filename'])
    with tf.gfile.GFile(camera_file_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image_format = 'jpeg'
    # image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()


    # Get the labels
    boxes, class_names, class_indices = get_label_data_2D_bounding_boxes(nusc, sample_token, sensor_channels=SENSOR_CHANNELS)
    
    # Box Dimensions
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    if len(boxes) > 0:
        boxes = np.array(boxes)
        xmins = boxes[:,0] # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = boxes[:,2] # List of normalized right x coordinates in bounding box
                    # (1 per box)
        ymins = boxes[:,1] # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = boxes[:,3] # List of normalized bottom y coordinates in bounding box
                    # (1 per box)

    # Classification data
    category_names = [c.encode('utf8') for c in class_names] # List of string class name of bounding box (1 per box)
    category_ids = class_indices # List of integer class id of bounding box (1 per box)

    feature_dict = {
        'image/calibration/intrinsic':
            dataset_util.float_list_feature(camera_intrinsic),
        'image/calibration/rotation':
            dataset_util.float_list_feature(camera_rotation),
        'image/calibration/translation':
            dataset_util.float_list_feature(camera_translation),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymaxs),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label': 
            dataset_util.int64_list_feature(category_ids),
        'radar/raw':
            dataset_util.float_list_feature(radar_data),
        'radar/calibration/rotation':
            dataset_util.float_list_feature(radar_rotation),
        'radar/calibration/translation':
            dataset_util.float_list_feature(radar_translation),
        'sample_token':
            dataset_util.bytes_feature(str(sample_token).encode('utf8')),

    }
        
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example


def get_sample_tokens(nusc, scene_indices=None):
    """
    This is a generator function.

    :param nusc: the database to get all tokens from
    :param scene_indices: [list of int or None] Determines 
        from which scenes the sample_tokens are going to be loaded

    :returns: [generator of str] all sample tokens from given scene_indices
    """
    if scene_indices is None:
        # we take all
        scene_indices = range(len(nusc.scene))

    # put all the scene tokens into a list
    scene_tokens = [nusc.scene[scene_index]['token']
                    for scene_index in scene_indices]

    # iterate over all the scenes and yield sample tokens
    for scene_token in scene_tokens:
        scene_rec = nusc.get('scene', scene_token)

        # iterate the samples in scene_rec
        sample_token = scene_rec['first_sample_token']
        while sample_token is not '':
            yield sample_token

            # get the next token
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']


def _create_tf_record_from_nuscenes_database(nusc, sample_tokens, output_path, num_shards):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
    :param nusc
        output_path: Path to output tf.Record file.
        num_shards: number of output file shards.
    """

    params = {
            'channels': CHANNELS,
            'image_size': IMAGE_SIZE,
            'radar_channel': RADAR_CHANNEL,
            'camera_channel': CAM_CHANNEL,
            'clear_radar': False,
            'clear_image': False,
            'dropout_chance': 0.0
            }

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_path, num_shards)

        # Iterate over all samples
        for idx, sample_token in enumerate(sample_tokens): 
            if idx % 100 == 0:
                tf.logging.info('On image %d of %d', idx, len(sample_tokens))

            _, tf_example = create_tf_example(nusc, sample_token, **params)
            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        
        tf.logging.info('Finished writing.')


def main(_):
    assert FLAGS.output_dir, '`output_dir` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'nuscenes_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'nuscenes_val.record')
    testdev_output_path = os.path.join(FLAGS.output_dir, 'nuscenes_testdev.record')

    # Creating the data generator
    nusc = NuScenes(version='v0.1', dataroot=NUSCENES_PATH, verbose=True)
    
    # Get all sample tokens
    val_scenes = 10
    train_sample_tokens = list(get_sample_tokens(nusc, range(len(nusc.scene)-val_scenes)))
    val_sample_tokens = list(get_sample_tokens(nusc, range(len(nusc.scene)-val_scenes, len(nusc.scene))))

    _create_tf_record_from_nuscenes_database(nusc, train_sample_tokens,
        train_output_path,
        num_shards=100)
    _create_tf_record_from_nuscenes_database(nusc, val_sample_tokens,
        val_output_path,
        num_shards=10)
    # _create_tf_record_from_nuscenes_database(
    #     FLAGS.testdev_annotations_file,
    #     FLAGS.test_image_dir,
    #     testdev_output_path,
    #     FLAGS.include_masks,
    #     num_shards=100)
    return 0


if __name__ == '__main__':
    main(None)
    # tf.app.run()
