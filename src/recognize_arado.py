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
import align.detect_face
from scipy import misc
import os
import sys
import math
import pickle
import cv2
from sklearn.svm import SVC

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)

            image_path = os.path.expanduser(args.image_path)
            print('Image filename: %s' % image_path)

            print('Image find faces')
            faces = findFaces(image_path, args)
            print('Found %s faces', len(faces))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            img = misc.imread(image_path)
            if img.ndim == 2:
                img = facenet.to_rgb(img)

            # Save the recognized faces as images and then load them for the model
            # TODO: Fiugure out why the model does not work if you use recognized faces directly without saving
            paths = []
            images = facenet.create_images_for(faces, img, False, False, args.image_size)
            for i in range(len(images)):
                path_i = "./temp/foo_"+`i`+".jpeg"
                misc.imsave(path_i, images[i])
                paths.append(path_i)

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(faces)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

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
                best_class = class_names[best_class_indices[i]]
                drawFaceRectangleToImage(img,faces[i], best_class)
                print('%4d  %s: %.3f' % (i, best_class, best_class_probabilities[i]))

            for i in range(len(images)):
                misc.imsave("foo_"+`i`+".jpeg", images[i])
            misc.imsave("out.jpeg", img)

            for p in paths:
                os.remove(p)




def findFaces(image_path, args):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join("./", filename + '_recognized.png')

    if not os.path.exists(output_filename):
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            print(errorMessage)
        else:
            if img.ndim < 2:
                print('Unable to align "%s"' % image_path)
                return
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:, :, 0:3]

            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                              factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces > 1:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    det_arr.append(np.squeeze(det))

                faces = []
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                    bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                    bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                    faces.append(bb)
                return faces

            else:
                print('Unable to align "%s"' % image_path)
    else:
        print("Output file exists")

def drawFaceRectangleToImage(img, face, class_name):
    cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (142, 194, 0), 2)
    cv2.putText(img, class_name, (face[0], face[3]+10), cv2.FONT_HERSHEY_PLAIN, 0.75, (255,255,255))

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
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=10)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
