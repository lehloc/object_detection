#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, url_for, send_from_directory, request, \
    render_template
import logging
import os
from werkzeug import secure_filename

# 20181125

import numpy as np
import os
import sys
import tensorflow as tf

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

# 20181125
# from gevent.pywsgi import WSGIServer

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 20181125 - AI Training
# What model to download.

MODEL_NAME = 'model'

# Path to frozen detection graph. This is the actual model that is used for the object detection.

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:

      # Get handles to input and output tensors

            tensor_dict = {}
            for key in ['detection_scores']:
                tensor_name = key + ':0'
                tensor_dict[key] = \
                    tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = \
                tf.get_default_graph().get_tensor_by_name('image_tensor:0'
                    )

      # Run inference

            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image,
                                   0)})

            output_dict['detection_scores'] = \
                output_dict['detection_scores'][0]

    return output_dict


def predict_number_object(image_path):
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np,
            detection_graph)

    count = 0
    for numb in output_dict['detection_scores']:
        if numb > 0.8:
            count += 1
            app.logger.info(numb)
    return count


# 20181125 - AI Training

# helper

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

# helper

@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST':
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['file']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info('saving {}'.format(saved_path))
        img.save(saved_path)
        predict_count = predict_number_object(saved_path)
        return str(predict_count)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)