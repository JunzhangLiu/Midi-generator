from PyQt5 import QtWidgets as qw
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication as qa
from PyQt5.QtGui import QIcon, QPixmap,QImage
import sys
import numpy as np
from model import Model
import tensorflow as tf
from PIL import Image
import os
from PIL.ImageQt import ImageQt
import pretty_midi
import pygame
from midi import *
from hyper_param import *
from saved_file_location import *
import random

class Slider_callback():
    def __init__(self,arg,fun):
        self.arg = arg
        self.fun = fun
    def __call__(self):
        return self.fun(self.arg)


class GUI(QMainWindow):
    def __init__(self,x,y,wid,ht,model,latent_dim=128,img_rescale=10):
        super(GUI,self).__init__()
        self.setGeometry(x,y,wid,ht)
        self.wid = wid
        self.ht=ht
        self.min_s = 10
        self.precision = 100000
        self.img_rescale = img_rescale

        #initialize latent vector
        self.std_range = 4
        self.transformation_mat = np.load(COMPONENTS_SAVE_LOCATION)
        self.components_std = np.load(COMPONENTS_STD_SAVE_LOCATION)
        self.components_mean = np.load(COMPONENTS_MEAN_SAVE_LOCATION)
        self.components = np.load(COMPONENTS_MEAN_SAVE_LOCATION)
        self.latent_vector= np.matmul(self.components,self.transformation_mat)
        
        #initialize music configuration
        self.play_speed = 0.25
        self.note_duration = 0.2
        self.num_sliders = 20
        self.model = model
        self.loaded = False
        self.is_button_press = False
        self.num_neighbor = 30
        self.threshold = 60

        self.init_ui()
        
    def init_ui(self):
        scale = 2
        self.get_labels(TIME_STEP // SECTION,88,scale)
        self.set_pix_map()
        self.init_button()
        self.init_sliders()
        self.show()
    def init_button(self):
        self.play = qw.QPushButton('play', self)
        self.play.move(800,700)
        self.play.clicked.connect(self.play_music)

        self.stop = qw.QPushButton('stop', self)
        self.stop.move(800,740)
        self.stop.clicked.connect(self.stop_music)

        self.reset = qw.QPushButton('reset', self)
        self.reset.move(800,780)
        self.reset.clicked.connect(self.reset_values)

        self.reset_speed = qw.QPushButton('reset speed', self)
        self.reset_speed.move(800,820)
        self.reset_speed.clicked.connect(self.reset_play_speed)

        self.findnn = qw.QPushButton('find nearest neighbor', self)
        self.findnn.move(800,860)
        self.findnn.clicked.connect(self.find_nearest)

        self.random = qw.QPushButton('random', self)
        self.random.move(1000,700)
        self.random.clicked.connect(self.random_song)

        
    def random_song(self):
        self.is_button_press = True
        for i in range(len(self.main_sliders)):
            self.main_sliders[i].setValue(random.randrange(-self.std_range*self.precision,self.std_range*self.precision))
        self.is_button_press = False
        self.set_pix_map()
    def set_pix_map(self,set_threshold=False):
        if set_threshold:
            midi_array = np.copy(self.midi_array)
        else:
            midi_array = self.model.generate_from_latent(np.expand_dims(self.latent_vector,axis=0))
            midi_array=midi_array.numpy()
            midi_array=np.squeeze(midi_array,axis=0)
            self.midi_array = np.copy(midi_array)
        midi_array[midi_array>=self.threshold/128]=1
        midi_array[np.logical_and(midi_array<self.threshold/128,midi_array>self.min_s/128)]=0.3
        midi_array[midi_array<=self.min_s/128]=0
        midi_array = np.flip(midi_array,axis=1)
        section_length = TIME_STEP // SECTION
        midi_array = np.reshape(midi_array,(SECTION,section_length,midi_array.shape[1]))
        scale = 2
        for i in range(SECTION):
            img = Image.fromarray(np.transpose(midi_array[i],axes=(1,0))*128)
            img = img.convert("RGBA")
            img = img.resize((int(img.size[0]*scale),int(img.size[1]*scale)),Image.NEAREST)
            img = ImageQt(img)
            pixmap = QPixmap.fromImage(img)
            self.labels[i].setPixmap(pixmap)

        
    def get_labels(self,width,height,scale):
        self.labels=[]
        for i in range(2):
            for j in range(SECTION//2):
                self.labels.append(qw.QLabel(self))
                self.labels[-1].move(650+j*(width*scale),i*(height*scale+10))
                self.labels[-1].resize(width*scale,height*scale)

    def init_sliders(self,gap = 40):
        self.main_sliders = []
        # self.fine_tune_sliders=[]
        self.callback_idx = []
        for i in range(2):
            for j in range(self.num_sliders):
                self.main_sliders.append(qw.QSlider(Qt.Horizontal,self))
                self.main_sliders[i*self.num_sliders+j].setGeometry(i*320, j*40, 300, 30)
                self.main_sliders[i*self.num_sliders+j].setMinimum(-self.std_range*self.precision)
                self.main_sliders[i*self.num_sliders+j].setMaximum(self.std_range*self.precision)
                self.main_sliders[i*self.num_sliders+j].setValue(0)
                self.callback_idx.append(i*self.num_sliders+j)
                self.main_sliders[i*self.num_sliders+j].valueChanged.connect(Slider_callback(i*self.num_sliders+j,self.val_change))
        
        self.threshold_slider = qw.QSlider(Qt.Horizontal,self)
        self.threshold_slider.setGeometry(800, 560, 300, 30)
        self.threshold_slider.setMinimum(self.min_s)
        self.threshold_slider.setMaximum(120)
        self.threshold_slider.setValue(60)
        self.threshold_slider.valueChanged.connect(self.set_threshold)

        self.speed = qw.QSlider(Qt.Horizontal,self)
        self.speed.setGeometry(800, 600, 300, 30)
        self.speed.setMinimum(1)
        self.speed.setMaximum(10)
        self.speed.setValue(5)
        self.speed.valueChanged.connect(self.set_speed)

        self.duration = qw.QSlider(Qt.Horizontal,self)
        self.duration.setGeometry(800, 640, 300, 30)
        self.duration.setMinimum(1)
        self.duration.setMaximum(20)
        self.duration.setValue(2)
        self.duration.valueChanged.connect(self.set_duration)

    def set_duration(self):
        self.note_duration = self.duration.value()/10


    def val_change(self,idx):
        # print(self.main_sliders[idx].value())
        deviation = (self.main_sliders[idx].value()/self.precision) * self.components_std[idx]
        self.components[idx]=self.components_mean[idx]+deviation
        # print("mean {:f}, dev {:f}, com {:f}, val {:f}".format(self.components_mean[idx],deviation,self.components[idx],self.main_sliders[idx].value()/self.precision))
        self.latent_vector=np.matmul(self.components,self.transformation_mat)
        if not self.is_button_press:
            self.set_pix_map()
    # def set_slider_param(self,param_name):
    #     self.slider_params[param_name] = self.sliders[param_name].value()

    def reset_play_speed(self):
        self.speed.setValue(5)
        self.play_speed=0.25

    def set_speed(self):
        self.play_speed = self.speed.value()/20

    def set_threshold(self):
        self.threshold=self.threshold_slider.value()
        self.set_pix_map(set_threshold=True)
    def play_music(self):
        array_to_midi(np.transpose(self.midi_array,axes=(1,0))*128,"./generated_music/foo",threshold = self.threshold_slider.value(),dur = self.note_duration,speed=self.play_speed)
        
        pygame.mixer.music.stop()
        pygame.mixer.music.load("./generated_music/foo.mid")
        pygame.mixer.music.play()
    def reset_values(self):
        self.is_button_press = True
        for i in range(2*self.num_sliders):
            # self.main_sliders[i].setValue(self.components_mean[i]*self.precision)
            self.main_sliders[i].setValue(0)
            # self.fine_tune_sliders[i].setValue(0)
        self.threshold_slider.setValue(60)
        self.is_button_press = False
        self.set_pix_map()
    def stop_music(self):
        pygame.mixer.music.stop()

    def find_nearest(self):
        if not self.loaded:
            print("data not loaded, start loading now")
            self.population = load_all()
            batch_size = 64
            sample_batches = self.population.shape[0]//batch_size
            population_latent_vec = []
            for i in range(sample_batches):
                print("sample:", i,end="\r")
                x = self.population[i*batch_size:(i+1)*batch_size]
                x = tf.cast(x,tf.float16)
                latent_vec_batch = self.model.get_latent_enc(x)
                latent_vec_batch=latent_vec_batch.numpy()
                population_latent_vec.append(latent_vec_batch)
            for i in range(sample_batches*batch_size,self.population.shape[0]):
                print("sample:", i,end="\r")
                x = self.population[i]
                x = tf.cast(x,tf.float16)
                latent_vec_batch=self.model.get_latent_enc(np.expand_dims(x,axis=0))
                latent_vec_batch=latent_vec_batch.numpy()
                population_latent_vec.append(latent_vec_batch)
            print("done")
            self.population_latent_vec = np.concatenate(population_latent_vec)
        self.loaded = True
        print("data loaded, find the nearest neighbor")
        dist = np.sqrt(np.sum((self.latent_vector-self.population_latent_vec)**2, axis=1))
        nn_idx = np.argsort(dist)[:self.num_neighbor]
        for idx,i in enumerate(nn_idx):
            print("nn {:d}, dist {:f}".format(idx,dist[i]))
            array_to_midi(np.transpose(self.population[i]*128,axes=(1,0)),"./nn/"+str(idx),speed=self.play_speed)
        print("found, midi pieces under ./nn/")

        

app = qa(sys.argv)
model = Model()
ckpt = tf.train.Checkpoint(model)
# path = os.path.join(os.path.dirname(__file__),"saved_model/save_mode/model")
path = os.path.join(os.path.dirname(__file__),SAVED_MODEL_LOCATION)
ckpt.read(path)#.assert_consumed()

# mixer config
freq = 44100  # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2  # 1 is mono, 2 is stereo
buffer = 1024   # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)

pygame.mixer.music.set_volume(0.8)

gui = GUI(200,200,1500,900,model)
sys.exit(app.exec_())