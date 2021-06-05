import numpy as np
import pretty_midi
import os
import functools
import tensorflow as tf
import tensorflow.keras as keras
import random
from PIL import Image
from hyper_param import *
class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
        ########################################
        # Todo: experiment l2 reg on encoder   #
        # Todo: Experiment lstm encoder        #
        # ###################################### 

        self.flatten = tf.keras.layers.Flatten()
        self.encoder = keras.Sequential([
                                keras.layers.Reshape((SECTION,TIME_STEP//SECTION*INPUT_DIM)),
                                keras.layers.TimeDistributed(keras.layers.Dense(2048)),
                                keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
                                keras.layers.ReLU(),
                                keras.layers.TimeDistributed(keras.layers.Dense(256)),
                                keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
                                keras.layers.ReLU(),
                                keras.layers.Flatten(),
                                keras.layers.Dense(1024),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),
                                keras.layers.Dense(LATENT_DIM),
                                keras.layers.BatchNormalization(),
                            ])
        #################################
        # todo: experiment lstm decoder #
        #################################
        self.decoder=keras.Sequential([
                                keras.layers.Dense(1024),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),
                                keras.layers.Dropout(DROP_OUT_RATE),

                                keras.layers.Dense(256*SECTION),
                                keras.layers.Reshape((SECTION,256)),
                                keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
                                keras.layers.ReLU(),
                                
                                keras.layers.Dropout(DROP_OUT_RATE),
                                
                                keras.layers.TimeDistributed(keras.layers.Dense(2048)),
                                keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
                                keras.layers.ReLU(),
                                
                                keras.layers.Dropout(DROP_OUT_RATE),

                                keras.layers.TimeDistributed(keras.layers.Dense((TIME_STEP//SECTION*INPUT_DIM),activation="sigmoid")),
                                keras.layers.Reshape((TIME_STEP,INPUT_DIM))
        ])
    def call(self,inputs, training=True):
        x = self.encoder(inputs, training=training)
        decoded = self.decoder(x,training=training)
        return decoded
    def sample(self,z_mu,z_sigma):
        eps = tf.random.normal(z_mu.shape,stddev=EPS_STD)
        return z_mu+tf.math.exp(z_sigma/2)*eps

    def generate_from_latent(self,z):
        decoded = tf.stop_gradient(self.decoder(z,training=False))
        return decoded

    def get_mu(self,inputs):
        x = tf.stop_gradient(self.encoder(inputs, training=False))
        return x
    def train_step(self,data):
        x = data
        with tf.GradientTape() as tape:
            encoded = self.encoder(x, training=True)
            # z_mu,z_sigma = self.latent_mu(encoded,training=True),self.latent_sigma(encoded,training=True) 
            # eps = tf.random.normal(tf.shape(z_mu),stddev=EPS_STD)
            # sample_z = z_mu+tf.math.exp(z_sigma/2)*eps
            y_pred = self.decoder(encoded,training=True)
            y_true = self.flatten(x)
            y_pred = self.flatten(y_pred)
            # reconstruction_loss = self.bce(y_true,y_pred)
            loss = tf.keras.losses.binary_crossentropy(y_true,y_pred)
            # kl_loss = -1 * tf.reduce_mean(1+z_sigma-tf.math.square(z_mu)-tf.math.exp(z_sigma),axis=-1)
            # loss = tf.reduce_mean(reconstruction_loss+kl_loss*KL_LOSS_WEIGHT)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        # self.kl_loss.update_state(kl_loss)
        # self.recon_loss.update_state(reconstruction_loss)
        return {"loss": self.loss_tracker.result()}#, "kl_loss": self.kl_loss.result(), "recon_loss": self.recon_loss.result()}