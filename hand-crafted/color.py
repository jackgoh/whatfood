#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Images binary classifier based on scikit-learn SVM classifier.
It uses the RGB color space as feature vector.
'''

from __future__ import division
from __future__ import print_function
from PIL import Image
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import svm
from sklearn import metrics
from StringIO import StringIO
from urlparse import urlparse
import urllib2
import sys
import os


def process_directory(directory):
    '''Returns an array of feature vectors for all the image files in a
    directory (and all its subdirectories). Symbolic links are ignored.

    Args:
      directory (str): directory to process.

    Returns:
      list of list of float: a list of feature vectors.
    '''
    training = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img_feature = process_image_file(file_path)
            if img_feature:
                training.append(img_feature)
    return training


def process_image_file(image_path):
    '''Given an image path it returns its feature vector.

    Args:
      image_path (str): path of the image file to process.

    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    image_fp = StringIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return process_image(image)
    except IOError:
        return None


def process_image_url(image_url):
    '''Given an image URL it returns its feature vector

    Args:
      image_url (str): url of the image to process.

    Returns:
      list of float: feature vector.

    Raises:
      Any exception raised by urllib2 requests.

      IOError: if the URL does not point to a valid file.
    '''
    parsed_url = urlparse(image_url)
    request = urllib2.Request(image_url)
    # set a User-Agent and Referer to work around servers that block a typical
    # user agents and hotlinking. Sorry, it's for science!
    request.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux ' \
            'x86_64; rv:31.0) Gecko/20100101 Firefox/31.0')
    request.add_header('Referrer', parsed_url.netloc)
    # Wrap network data in StringIO so that it looks like a file
    net_data = StringIO(urllib2.build_opener().open(request).read())
    image = Image.open(net_data)
    return process_image(image)


def process_image(image, blocks=4):
    '''Given a PIL Image object it returns its feature vector.

    Args:
      image (PIL.Image): image to process.
      blocks (int, optional): number of block to subdivide the RGB space into.

    Returns:
      list of float: feature vector if successful. None if the image is not
      RGB.
    '''
    if not image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for pixel in image.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]


def show_usage():
    '''Prints how to use this program
    '''
    print("Usage: %s [class A images directory] [class B images directory]" %
            sys.argv[0])
    sys.exit(1)


def train(training_path_a, training_path_b, print_metrics=True):
    '''Trains a classifier. training_path_a and training_path_b should be
    directory paths and each of them should not be a subdirectory of the other
    one. training_path_a and training_path_b are processed by
    process_directory().

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
      print_metrics  (boolean, optional): if True, print statistics about
        classifier performance.

    Returns:
      A classifier (sklearn.svm.SVC).
    '''
    if not os.path.isdir(training_path_a):
        raise IOError('%s is not a directory' % training_path_a)
    if not os.path.isdir(training_path_b):
        raise IOError('%s is not a directory' % training_path_b)
    training_a = process_directory(training_path_a)
    training_b = process_directory(training_path_b)
    # data contains all the training data (a list of feature vectors)
    data = training_a + training_b
    # target is the list of target classes for each feature vector: a '1' for
    # class A and '0' for class B
    target = [1] * len(training_a) + [0] * len(training_b)
    # split training data in a train set and a test set. The test set will
    # containt 20% of the total
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,
            target, test_size=0.20)
    # define the parameter search space
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],
            'gamma': [0.01, 0.001, 0.0001]}
    # search for the best classifier within the search space and return it
    clf = grid_search.GridSearchCV(svm.SVC(), parameters).fit(x_train, y_train)
    classifier = clf.best_estimator_
    if print_metrics:
        print()
        print('Parameters:', clf.best_params_)
        print()
        print('Best classifier score')
        print(metrics.classification_report(y_test,
            classifier.predict(x_test)))
    return classifier


def main(training_path_a, training_path_b):
    '''Main function. Trains a classifier and allows to use it on images
    downloaded from the Internet.

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
    '''
    print('Training classifier...')
    classifier = train(training_path_a, training_path_b)
    while True:
        try:
            print("Input an image url (enter to exit): "),
            image_url = raw_input()
            if not image_url:
                break
            features = process_image_url(image_url)
            print(classifier.predict(features))
        except (KeyboardInterrupt, EOFError):
            break
        except:
            exception = sys.exc_info()[0]
            print(exception)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        show_usage()
    main(sys.argv[1], sys.argv[2])
