from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
from keras.callbacks import Callback
from keras.optimizers import Adam as Adam

import os
import sys
import zipfile


from keras import backend as K
from keras.utils.generic_utils import CustomObjectScope
import math
import shutil

test_dir = sys.argv[1]

zip_ref = zipfile.ZipFile(sys.argv[1], 'r')
zip_ref.extractall("./test/test1")
zip_ref.close()

file = open("hyperparameter.txt",'r')

i = 0
img_width = 0
for val in file:
    if i == 0:
        val.split()
        alpha_1 = float(val[0])
        i = i+1
    else:
        img_width = int(val)
img_height = img_width

test_data_dir = "./test/test1"
model_path = sys.argv[2]
img_channels = 3 
batch_size = 4

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(test_data_dir,
                                            target_size=(img_height, img_width),
                                            batch_size=batch_size,
                    					    shuffle = False,
                                            class_mode='categorical')

listing=os.listdir(test_data_dir)
number_of_test_images = 0
for folder in listing:
    inner = os.listdir(test_data_dir + '/' + folder)
    number_of_test_images += len(inner)

test_steps = math.ceil(number_of_test_images/batch_size)
with CustomObjectScope({'relu6': applications.mobilenet.relu6,'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
    	model = load_model(model_path)

print("Waiting For Result.........")
history = model.evaluate_generator(test_set, steps=test_steps)
print("*****************")
print ("Test Accuracy: "  + str(history[1]))
print("*****************")
shutil.rmtree("./test")