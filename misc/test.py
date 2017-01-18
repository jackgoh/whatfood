from PIL import Image
from skimage import color, exposure
from skimage.feature import hog
from skimage.transform import resize
from scipy.misc import imshow
from glob import glob
import skimage.io as io
from PIL import Image
import numpy as np
import scipy.misc
import cv2
import random
import matplotlib.pyplot as plt
import os
import gzip
import hickle as hkl
import imutils

#Directory containing images you wish to convert
input_dir = "../dataset/food100"
dataFile = '../dataset/rgb_myfood100.hkl'
directories = os.listdir(input_dir)

index = 0
index2 = 0
labels = []
training = []
blocks = 4

print directories
for folder in directories:
  #Ignoring .DS_Store dir
  if folder == '.DS_Store':
    pass

  else:
    print index, folder

    folder2 = os.listdir(input_dir + '/' + folder)


    for image in folder2:
      if image == ".DS_Store":
        pass

      else:
        index2 += 1

        #image = cv2.imread(input_dir+"/"+folder+"/"+image)
        image = Image.open(input_dir+"/"+folder+"/"+image)
        #hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        #hist = cv2.calcHist([hsv], [0, 1, 2], None, (8,8,8), [0, 180, 0, 256, 0, 256])

        feature = [0] * blocks * blocks * blocks
        pixel_count = 0

        for pixel in image.getdata():
            ridx = int(pixel[0]/(256/blocks))
            gidx = int(pixel[1]/(256/blocks))
            bidx = int(pixel[2]/(256/blocks))
            idx = ridx + gidx * blocks + bidx * blocks * blocks
            feature[idx] += 1
            pixel_count += 1

        try:

          if index2 != 1:
            training.append([x/pixel_count for x in feature])

          elif index2 == 1:
            training.append([x/pixel_count for x in feature])

          if index == 0 and index2 == 1:
            index_array = np.array([[index]])
            labels.append(index)

          else:
            new_index_array = np.array([[index]], np.int8)
            index_array = np.append(index_array, new_index_array, 0)
            labels.append(index)

        except Exception as e:
          print e
          print "Defect image: " + image
          #os.remove(input_dir+"/"+folder+"/"+image)
    index += 1

print index
print "Total train set: ", len(out)
print "Total train label: " ,len(index_array)

data={'trainFeatures':training, 'trainLabels':  index_array, 'labels': labels }
hkl.dump(data, dataFile, mode='w')
print("Dumped hkl into "+dataFile)
