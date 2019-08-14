###############################################################################
# Author: Benny Avelin                                                        #
# Date: 2019-04-15                                                            #
# Project: DeepLimit                                                          #
#                                                                             #
# Purpose: This code file will run a cross validated test of a finite layer   #
# neural ODE.                                                                 #
# The problem that it is trying to solve is a classification problem of       #
# two-ring-type.                                                              #
# In polar form we have one ring that looks like                              #
# r = r_0(2+\cos(5*\theta))                                                   #
#                                                                             #
###############################################################################


###############################################################################
###############################################################################
from keras.layers import (Dense, 
                         Add, 
                         Lambda, 
                         Input)

from keras import optimizers, regularizers

from keras.models import Model

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

import keras.utils
import keras.backend as K

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
    lr = 1e-1
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
          augmented_dimensions=0,
          callbacks=None):
    """training a finite layer network on x_train, use x_test,y_test to validate
    # Arguments
        x_train (n_samples,2): training data
        y_train (n_samples): the corresponding classes
        x_test (n_test,2): test data
        y_test (n_test): the corresponding classes
        N (int): the number of finite neural ODE steps
        epochs (int): the number of epochs to run
        gpus (int): how many gpus should be used for training
        batch_size (int): the batch size
        augmented_dimensions (int): number of zero dimensions to add to input
        callbacks (list): the list of Keras callbacks to used during training
    # Returns
        A tuple containing the test loss and test accuracy
    """
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    ## Define the layers
    
    x0 = Input(shape=(x_train.shape[1],), name='Input') # Input shape

    hidden_size = 16
    
    nn_block_in = Dense(hidden_size, 
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.001),
                        name='nn_block_in')
                     
    nn_block_out = Dense(x_train.shape[1], 
                         activation=None,
                         kernel_regularizer=regularizers.l2(0.001),
                         name='nn_block_out')
                                           
    linear_output = Dense(units=2, 
                          activation='softmax', 
                          kernel_initializer='random_normal',
                          name='linear_output')


    ## Put the network together
    
    x1 = x0
    
    ## Finite step Neural ODE
    for i in range(N):
        x1 = ((Add()([Lambda(lambda x:x/N)(nn_block_out(nn_block_in(x1))),x1])))
    
    ## Linear classification layer
    y = linear_output(x1)

    ## Define the model and if multi_gpu, copy to all gpus.
    model = Model(inputs=x0, outputs=y)
    if (gpus > 1):
        model = keras.utils.multi_gpu_model(model, 
                                            gpus=gpus, 
                                            cpu_merge=True, 
                                            cpu_relocation=True)
                                            
    model.compile(optimizer=optimizers.SGD(lr=lr_schedule(0)),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    ## This part contains the data normalization
    
    subtract_mean=True
    
    if subtract_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    
    # Run training
    model.fit(x_train, 
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
    
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
    # Saves the trained weights
    save_dir = os.path.join(os.getcwd(), 'AnnuliExperiment')
    model_name = 'Annulus_%d_kfold_%d_model_%d.h5' % (augmented_dimensions,N,randint(0,10000))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    
    model.save_weights(filepath)
    
    return scores
    
def cross_val(k=3, 
              depth=1, 
              epochs=1, 
              gpus=1, 
              batch_size=1, 
              augmented_dimensions=0,
              callbacks=None):
    """Does a k fold split and trains a model for each split. Saves the results of the k-fold as a csv.
    # Arguments
        k (int): The number of folds
        depth (int): The number of finite neural ODE steps
        epochs (int): Epochs of each fold
        gpus (int): The number of gpus used for training
        batch_size (int): Batch size
        augmented_dimensions (int): number of zero dimensions to add to input
        callbacks (list): List of Keras callbacks to be used during training
    # Returns
        None
    """
    ## Load the data
    X,y = generate_data(r1=1,
                        r2=1.5,
                        r3=3,
                        size=50000,
                        num_augmented_dimensions=augmented_dimensions)
    
    ## Construct a k-fold split
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    cvscores = []
    for train_ind, test_ind in kfold.split(X, y):
        cvscores.append(
            train(
                X[train_ind], # The training data for this fold
                y[train_ind], # The training data for this fold
                X[test_ind],  # The test data for this fold
                y[test_ind],  # The test data for this fold
                depth,
                epochs,
                gpus,
                batch_size*gpus,    # Each gpu gets batch_size samples
                augmented_dimensions,
                callbacks))
                
    print("Trained with N=%d, epochs=%d, batch_size=%d, k=%d" % (depth,epochs,batch_size,k))
    print(np.array(cvscores)[:,0])
    print("Average accuracy over %d-fold: %.4f" % (k,np.mean(np.array(cvscores)[:,1])))
    
    ## Create a dataframe with the results and save to csv
    import pandas as pd
    df = pd.DataFrame(np.array(cvscores)[:,0].reshape(1,-1))
    df.to_csv('AnnuliExperiment/Annulus_%d_%d.csv' % (augmented_dimensions,depth),header=False,index=False)

def generate_data(r1=1,r2=2,r3=3,size=100,num_augmented_dimensions=0):
    """
    Generates points from two classes consisting of a starlike set inside
    a starlike ring. This is fairly difficult to classify as the sets have
    corners where they are very close and it requires fairly complex estimators
    # Arguments
        r1 (float32): The size of the inner starlike set
        r2 (float32): The inner radii of the outer starlike ring
        r3 (float32): The outer radii of the outer starlike ring
        size (int): The number of points to sample, it will be rounded down to an even number
        num_augmented_dimensions (int): The number of extra dimensions to fill with zeros
    """
    # Radii drawn uniformly at random
    # r1_rand \in [0,r_1]
    r1_rand = np.random.uniform(size=(size//2,1))*r1
    # r2_3_rand \in [r2,r3]
    r2_3_rand = np.random.uniform(size=(size//2,1))*(r3-r2)+r2
    
    # Angles drawn uniformly at random from [-\pi,\pi]
    angle_rand_1 = np.random.uniform(size=size//2)*np.pi*2-np.pi
    angle_rand_2 = np.random.uniform(size=size//2)*np.pi*2-np.pi
    
    # The inner set consists of the following set (in polar form)
    # r = \rho * (2+cos(\theta)), \rho \in [0,r1]
    # \theta = \theta
    inner_set = (r1_rand
                    *(2+np.cos(5*angle_rand_1.reshape(-1,1)))
                    *np.stack([np.cos(angle_rand_1),np.sin(angle_rand_1)],axis=1))
            
    # The outer ring consists of the following set (in polar form)
    # r = \rho * (2+cos(\theta)), \rho \in [r2,r3]
    # \theta = \theta
    outer_set = ((2+np.cos(5*angle_rand_2.reshape(-1,1)))
                    *r2_3_rand
                    *np.stack([np.cos(angle_rand_2),np.sin(angle_rand_2)],axis=1))

            
    X = np.concatenate([inner_set,outer_set])
    
    # If we should augment the dimensions, let us just add zeros
    if (num_augmented_dimensions > 0):
        X = np.hstack([X,np.zeros((len(X),num_augmented_dimensions))])
        
    # The class labels
    y = np.concatenate([np.zeros(size//2),np.ones(size//2)])
    return X,y    

def main(argv):
    depth = int(argv[0])
    epochs = int(argv[1])
    gpus = int(argv[2])
    k = int(argv[3])
    augmented_dimensions = int(argv[4])
    
    print("Running with N:%d, epochs:%d, gpus:%d, k:%d, augmented_dimension:%d" 
        % (depth,epochs,gpus,k,augmented_dimensions))

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [lr_reducer, lr_scheduler]
    cross_val(k,depth,epochs,gpus,128,augmented_dimensions,callbacks)
    K.clear_session()    

if __name__ == "__main__":
   main(sys.argv[1:])
