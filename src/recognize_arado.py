# MIT License
# 
# Copyright (c) 2016 Jarno Rantala
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)

            image_path = os.path.expanduser(args.image_path)
            print('Image filename: %s' % image_path)

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            images = facenet.load_data([image_path], False, False, args.image_size)
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            # Classify images
            print('Running classifier')
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            print('Loaded classifier model from file "%s"' % classifier_filename_exp)
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str,
        help='Path to the image to recognize.')
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
        default='../models/20170512-110547/20170512-110547.pb')
    parser.add_argument('--classifier_filename',
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.',
        default='../models/arado_classifier.pkl')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
