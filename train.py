
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

import math
from keras import backend as K
from keras.utils.generic_utils import CustomObjectScope
import shutil


def training_data(train_data_dir,img_height, img_width,batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255,  # normalize to ensure faster convergence and avoid overfitting
                                       rotation_range=180,  # Augmentations
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=[0.8, 1.5],
                                       fill_mode='nearest')  # fill up image pixels with nearest neighbors
    training_set = train_datagen.flow_from_directory(train_data_dir,
                                                     target_size=(img_height, img_width),
                                                     batch_size=batch_size,
                                                     class_mode='categorical')
    listing = os.listdir(train_data_dir)
    number_of_train_images = 0
    for folder in listing:
        inner = os.listdir(train_data_dir + '/' + folder)
        number_of_train_images += len(inner)
    return training_set,number_of_train_images




def validate_data(validation_data_dir,img_height, img_width,batch_size):
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_set = validation_datagen.flow_from_directory(validation_data_dir,
                                                            target_size=(img_height, img_width),
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            class_mode='categorical')
    listing = os.listdir(validation_data_dir)
    number_of_test_images = 0
    for folder in listing:
        inner = os.listdir(validation_data_dir + '/' + folder)
        number_of_test_images += len(inner)

    return validation_set,number_of_test_images




def model_creator(alpha_1, img_width, img_height, img_channels,nb_classes):
    model = applications.mobilenet.MobileNet(alpha=alpha_1, weights='imagenet', include_top=False,
                                             input_shape=(img_width, img_height, img_channels))
    # MobileNet => Built in Keras model for mobile application. Light Deep CNN architecture
    # alpha => width multiplier. Hyper Parameter. Can be varied from 1.0 to 0.25 [1.0, 0.75, 0.5, 0.25]
    # weights = "imagenet" => Pre-trained weights
    # include_top = False => discarded top layer of MobileNet, because that is used for 1000 categories of imageNet dataset
    # image_width, image_Height, image_channels => same as before


    x = model.output
    x = Flatten()(
        x)  # output of imageNet feature extractor is 2-dimensional. So, we flatten it first to generate 1D array
    x = Dense(1024, activation="relu")(x)  # Relu => negative becomes zero
    x = Dropout(0.3)(x)  # Dropout 30% of total nodes
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(output_dim=nb_classes, activation="softmax")(x)  # returns a probability vector of 8 classes
    model_final = Model(input=model.input, output=predictions)

    adam = Adam(lr=1e-4)  # control learning rate
    model_final.compile(optimizer=adam, loss='categorical_crossentropy',
                        metrics=['accuracy'])  # loss function for categorical data
    return model_final


if __name__ == "__main__":

    zip_ref = zipfile.ZipFile(sys.argv[1], 'r')
    zip_ref.extractall("./test/train")
    zip_ref.close()

    str = sys.argv[2]
    file = open(str[2:],'r')

    i = 0
    img_width = 0
    for val in file:
        if i == 0:
            val.split()
            alpha_1 = float(val[0])
            i = i+1
        else:
            img_width = int(val)
    img_height = img_width #image resolution hyper parameter
    train_data_dir = "./test/train" #link of train data directory
    model_path = sys.argv[3] #model saving path

    img_channels = 3 #RGB
    nb_classes = 8 #final categories
    batch_size = 8 #how many images at a time, can be treated as a Hyper Parameter

    nb_epoch = 5 #iterations


    print (alpha_1)
    print (img_width)
    print (sys.argv[1])
    print (model_path)

    training_set,number_of_train_images = training_data(train_data_dir,img_height, img_width,batch_size)

    steps_per_epoch = math.ceil(number_of_train_images/batch_size)




    model_final = model_creator(alpha_1, img_width, img_height, img_channels,nb_classes)


    history = model_final.fit_generator(training_set,
                        steps_per_epoch = steps_per_epoch,
                        epochs=nb_epoch,
                        initial_epoch=0)

    model_final.save(model_path)

    shutil.rmtree("./test")