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
        self.train_loss_tracker = keras.metrics.Mean(name="loss")
        self.flatten = tf.keras.layers.Flatten()
        self.encoder = keras.Sequential([
                                keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True)),
                                keras.layers.Bidirectional(keras.layers.LSTM(32,return_sequences=True)),
                                keras.layers.Bidirectional(keras.layers.LSTM(1,return_sequences=True)),
                                keras.layers.Flatten(),
                                keras.layers.Dense(1024),#,kernel_regularizer=tf.keras.regularizers.L2()),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),
                                keras.layers.Dense(LATENT_DIM),#,kernel_regularizer=tf.keras.regularizers.L2()),
                                keras.layers.BatchNormalization(),
                            ])
        self.decoder=keras.Sequential([
                                keras.layers.Dense(16*TIME_STEP),#,kernel_regularizer=tf.keras.regularizers.L2()),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),
                                keras.layers.Dropout(DROP_OUT_RATE),

                                keras.layers.Reshape((TIME_STEP,16)),

                                keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True)),
                                keras.layers.Dropout(DROP_OUT_RATE),
                                
                                keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True)),
                                keras.layers.Dropout(DROP_OUT_RATE),
                                keras.layers.LSTM(88,return_sequences=True),
        ])
    def call(self,inputs, training=True):
        x = self.encoder(inputs, training=training)
        decoded = self.decoder(x,training=training)
        return decoded

    # def sample(self,z_mu,z_sigma):
    #     eps = tf.random.normal(z_mu.shape,stddev=EPS_STD)
    #     return z_mu+tf.math.exp(z_sigma/2)*eps

    def generate_from_latent(self,z):
        decoded = tf.stop_gradient(self.decoder(z,training=False))
        return decoded

    def get_latent_enc(self,inputs):
        x = tf.stop_gradient(self.encoder(inputs, training=False))
        return x
    def train_step(self,data):
        # x = data
        
        x = tf.cast(data,tf.float16)
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
        self.train_loss_tracker.update_state(loss)
        # self.kl_loss.update_state(kl_loss)
        # self.recon_loss.update_state(reconstruction_loss)
        return {"loss": self.train_loss_tracker.result()}#, "kl_loss": self.kl_loss.result(), "recon_loss": self.recon_loss.result()}
