###############################################################################
# Author: Benny Avelin                                                        #
# Date: 2019-04-15                                                            #
# Project: DeepLimit                                                          #
#                                                                             #
# Purpose: This code file will run a cross validated test of a finite layer   #
# neural ODE with CNN structure.                                              #
###############################################################################


###############################################################################
###############################################################################
from keras.layers import (Dense, 
                          Conv2D, 
                          AvgPool2D, 
                          Add, 
                          Lambda, 
                          Input, 
                          Reshape, 
                          BatchNormalization)

from keras import optimizers

from keras.models import Model

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

import keras.utils
import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import cifar10

import numpy as np
import sys
import os
import random

###############################################################################
###############################################################################

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def train(x_train, 
          y_train, 
          x_test, 
          y_test, 
          N=1, 
          epochs=1, 
          gpus=1, 
          batch_size=1, 
          callbacks=None):
    """training a finite layer network on x_train, use x_test,y_test to validate
    # Arguments
        x_train (n_samples, 32, 32, 3): training data
        y_train (n_samples): the corresponding classes
        x_test (n_test,32,32,3): test data
        y_test (n_test): the corresponding classes
        N (int): the number of finite neural ODE steps
        epochs (int): the number of epochs to run
        gpus (int): how many gpus should be used for training
        batch_size (int): the batch size
        callbacks (list): the list of Keras callbacks to used during training
    # Returns
        A tuple containing the test loss and test accuracy
    """
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    ## Define the layers
    
    x0 = Input(shape=(32,32,3,)) # Input shape from cifar10

    cnn_in = Conv2D(256, 
                        kernel_size=3, 
                        activation='relu',
                        strides=(2,2), 
                        padding='same')
                        
    nn_block_in = Conv2D(256, 
                     kernel_size=3, 
                     activation='relu',
                     strides=(1,1), 
                     padding='same')
                     
    nn_block_out = Conv2D(256, 
                      kernel_size=3, 
                      activation=None,
                      strides=(1,1), 
                      padding='same')
                      
    cnn_out = Conv2D(64, 
                     kernel_size=3, 
                     activation='relu',
                     strides=(4,4), 
                     padding='same')
                     
    linear_output = Dense(units=10, 
                          activation='softmax', 
                          kernel_initializer='random_normal')


    ## Put the network together
    
    x1 = cnn_in(x0)
    x2 = BatchNormalization()(x1)

    ## Finite step Neural ODE
    for i in range(N):
        ## Single resnet type block
        x2 = ((Add()([Lambda(lambda x:x/N)(nn_block_out(nn_block_in(x2))),x2])))
    
    ## Final downsizing and projections
    x3 = BatchNormalization()(x2)
    x4 = cnn_out(x3)
    x5 = BatchNormalization()(x4)
    x6 = AvgPool2D(4)(x5)
    #Flattening the output from AvgPool2D
    x7 = Reshape((64,))(x6)
    
    ## Linear classification layer
    y = linear_output(x7)

    ## Define the model and if multi_gpu, copy to all gpus.
    model = Model(inputs=x0, outputs=y)
    if (gpus > 1):
        model = keras.utils.multi_gpu_model(model, 
                                            gpus=gpus, 
                                            cpu_merge=True, 
                                            cpu_relocation=True)
                                            
    model.compile(optimizer=optimizers.Adam(lr=lr_schedule(0)),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    ## This part contains the data augmentation for training
    ## The code is taken from the Keras documentation for the Cifar10 example
    ## It does ZCA whitening and does horizontal flipping, width_shifts
    ## and height_shifts
    
    subtract_pixel_mean=True
    data_augmentation=True
    
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    
    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=1, use_multiprocessing=True,
                            callbacks=callbacks, steps_per_epoch=x_train.shape[0]//batch_size)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
    # Saves the trained weights
    save_dir = os.path.join(os.getcwd(), 'Cifar10Experiment')
    model_name = 'cifar10_kfold_%d_%s_model.h5' % (N,str(random.randint(0,10000)))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    
    model.save_weights(filepath)
    
    return scores
    
def cross_val(k=3, depth=1, epochs=1, gpus=1, batch_size=1,callbacks=None):
    """Does a k fold split and trains a model for each split. Saves the results of the k-fold as a csv.
    # Arguments
        k (int): The number of folds
        depth (int): The number of finite neural ODE steps
        epochs (int): Epochs of each fold
        gpus (int): The number of gpus used for training
        batch_size (int): Batch size
        callbacks (list): List of Keras callbacks to be used during training
    # Returns
        None
    """
    ## Load the data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    ## Reshape to right format and rescale to be between 0 and 1
    x_train = x_train.reshape((50000,32,32,3))/255
    x_test = x_test.reshape((10000,32,32,3))/255
    
    ## Construct a k-fold split
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    cvscores = []
    for train_ind, test_ind in kfold.split(x_train, y_train):
        cvscores.append(
            train(
                x_train[train_ind], # The training data for this fold
                y_train[train_ind], # The training data for this fold
                x_train[test_ind],  # The test data for this fold
                y_train[test_ind],  # The test data for this fold
                depth,
                epochs,
                gpus,
                batch_size*gpus,    # Each gpu gets batch_size samples
                callbacks))
                
    print("Trained with N=%d, epochs=%d, batch_size=%d, k=%d" % (depth,epochs,batch_size,k))
    print(np.array(cvscores)[:,1])
    print("Average accuracy over %d-fold: %.4f" % (k,np.mean(np.array(cvscores)[:,1])))
    
    ## Create a dataframe with the results and save to csv
    import pandas as pd
    df = pd.DataFrame(np.array(cvscores)[:,1].reshape(1,-1))
    df.to_csv('Cifar10Experiment/%d.csv' % depth,header=False,index=False)
    
def main(argv):
    depth = int(argv[0])
    epochs = int(argv[1])
    gpus = int(argv[2])
    k = int(argv[3])

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [lr_reducer, lr_scheduler]
    cross_val(k,depth,epochs,gpus,128,callbacks)
    K.clear_session()    

if __name__ == "__main__":
   main(sys.argv[1:])