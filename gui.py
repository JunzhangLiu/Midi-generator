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
import random
######################################
#   todo: clean up this mess!!!      #
######################################

class Slider_callback():
    def __init__(self,arg,fun):
        self.arg = arg
        self.fun = fun
    def __call__(self):
        return self.fun(self.arg)

class Mixer(QMainWindow):
    def __init__(self,x,y,wid,ht,model,latent_dim=128,img_rescale=10):
        super(Mixer,self).__init__()
        self.setGeometry(x,y,wid,ht)
        self.setWindowTitle("Hello world!")
        self.wid = wid
        self.ht=ht
        self.min_s = 10
        self.precision = 100000
        self.cover_std = 3
        self.transformation_mat = np.load(COMPONENTS_SAVE_LOCATION)
        self.components_std = np.load(COMPONENTS_STD_SAVE_LOCATION)
        self.components_mean = np.load(COMPONENTS_MEAN_SAVE_LOCATION)
        self.play_speed = 30
        self.data_std = np.load(DATA_STD_SAVE_LOCATION)
        self.data_mean = np.load(DATA_MEAN_SAVE_LOCATION)

        self.default_value = np.copy(self.components_mean)
        self.latent_vector= np.matmul(self.components_mean,self.transformation_mat)
        
        self.note_duration = 0.2

        self.num_sliders = 20
        self.model = model
        self.img_rescale = img_rescale
        self.population = None
        self.population_mu = None
        self.loaded = False
        self.k = 30
        self.s = 60

        self.init_ui()
        
    def init_ui(self):
        seq = self.model.generate_from_latent(np.expand_dims(self.latent_vector,axis=0))
        seq=seq.numpy()
        seq=np.squeeze(seq,axis=0)
        self.music = np.copy(seq)
        seq[seq>=self.s/128]=1
        seq[np.logical_and(seq<self.s/128, seq>self.min_s/128)]=0
        seq[seq<=self.min_s/128]=0
        seq = np.flip(seq,axis=1)
        section_length = TIME_STEP // SECTION
        seq = np.reshape(seq,(SECTION,section_length,seq.shape[1]))
        scale = 2
        self.get_labels(seq.shape[1],seq.shape[2],scale)
        for i in range(SECTION):
            img = Image.fromarray(np.transpose(seq[i],axes=(1,0))*128)
            img = img.convert("RGBA")
            img = img.resize((int(img.size[0]*scale),int(img.size[1]*scale)),Image.NEAREST)
            img = ImageQt(img)
            pixmap = QPixmap.fromImage(img)
            # pixmap = pixmap.scaledToWidth(256)
            self.labels[i].resize(pixmap.width(),pixmap.height())
            self.labels[i].setPixmap(pixmap)
            self.labels[i].setScaledContents(True)
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
    
        # qim = QImage(img.tobytes("raw"),img.size[0], img.size[1], QImage.Format_ARGB32)
        # pixmap = QPixmap.fromImage(qim)
        # pixmap = pixmap.scaledToWidth(256)
            # self.labels[idx].resize(pixmap.width()*self.img_rescale,pixmap.height()*self.img_rescale)
        self.get_sliders()
        self.show()
        
    def random_song(self):
        for i in range(len(self.main_sliders)):
            self.main_sliders[i].setValue(random.randrange(-self.cover_std*self.precision,self.cover_std*self.precision))

    def set_pix_map(self,scaled=False):
        if scaled:
            seq = np.copy(self.music)
        else:
            seq = self.model.generate_from_latent(np.expand_dims(self.latent_vector,axis=0))
            seq=seq.numpy()
            seq=np.squeeze(seq,axis=0)
            self.music = np.copy(seq)
        seq[seq>=self.s/128]=1
        seq[np.logical_and(seq<self.s/128,seq>self.min_s/128)]=0.3
        seq[seq<=self.min_s/128]=0
        seq = np.flip(seq,axis=1)
        section_length = TIME_STEP // SECTION
        seq = np.reshape(seq,(SECTION,section_length,seq.shape[1]))
        scale = 2
        for i in range(SECTION):
            img = Image.fromarray(np.transpose(seq[i],axes=(1,0))*128)
            img = img.convert("RGBA")
            img = img.resize((int(img.size[0]*scale),int(img.size[1]*scale)),Image.NEAREST)
            img = ImageQt(img)
            pixmap = QPixmap.fromImage(img)
            # pixmap = pixmap.scaledToWidth(256)
            # self.labels[i].resize(pixmap.width(),pixmap.height())
            self.labels[i].setPixmap(pixmap)
            # self.labels[i].setScaledContents(True)

        
    def get_labels(self,width,height,scale):
        self.labels=[]
        idx = 0
        for i in range(2):
            for j in range(SECTION//2):
                self.labels.append(qw.QLabel(self))
                self.labels[idx].move(650+j*(width*scale+10),i*(height*scale+10))
                idx+=1
    def get_sliders(self,gap = 40):
        self.main_sliders = []
        # self.fine_tune_sliders=[]
        self.callback_idx = []
        for i in range(2):
            for j in range(self.num_sliders):
                self.main_sliders.append(qw.QSlider(Qt.Horizontal,self))
                self.main_sliders[i*self.num_sliders+j].setGeometry(i*320, j*40, 300, 30)
                # self.main_sliders[i*self.num_sliders+j].setMinimum((self.components_mean[i*self.num_sliders+j]-self.cover_std*self.components_std[i*self.num_sliders+j])*self.precision)
                # self.main_sliders[i*self.num_sliders+j].setMaximum((self.components_mean[i*self.num_sliders+j]+self.cover_std*self.components_std[i*self.num_sliders+j])*self.precision)
                self.main_sliders[i*self.num_sliders+j].setMinimum(-self.cover_std*self.precision)
                self.main_sliders[i*self.num_sliders+j].setMaximum(self.cover_std*self.precision)
                self.main_sliders[i*self.num_sliders+j].setValue(0)
                self.callback_idx.append(i*self.num_sliders+j)
                self.main_sliders[i*self.num_sliders+j].valueChanged.connect(Slider_callback(i*self.num_sliders+j,self.val_change))

                
                # self.fine_tune_sliders.append(qw.QSlider(Qt.Horizontal,self))
                # self.fine_tune_sliders[i].setGeometry(320, i*40, 300, 30)
                # self.fine_tune_sliders[i].setMinimum(-0.5*self.precision*self.components_std[i])
                # self.fine_tune_sliders[i].setMaximum(0.5*self.precision*self.components_std[i])
                # self.fine_tune_sliders[i].setValue(0)
                # self.callback_idx.append(i)
                # self.fine_tune_sliders[i].valueChanged.connect(Slider_callback(i,self.val_change))

        self.scale = qw.QSlider(Qt.Horizontal,self)
        self.scale.setGeometry(800, 560, 300, 30)
        self.scale.setMinimum(self.min_s)
        self.scale.setMaximum(120)
        self.scale.setValue(60)
        self.scale.valueChanged.connect(self.note_scale)

        self.speed = qw.QSlider(Qt.Horizontal,self)
        self.speed.setGeometry(800, 600, 300, 30)
        self.speed.setMinimum(-40)
        self.speed.setMaximum(-10)
        self.speed.setValue(-30)
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
        # print((self.main_sliders[idx].value()+self.fine_tune_sliders[idx].value())/self.precision)
        # self.components_mean[idx]=(self.main_sliders[idx].value()+self.fine_tune_sliders[idx].value())/self.precision
        
        self.components_mean[idx]=self.default_value[idx]+((self.main_sliders[idx].value())/self.precision) * self.components_std[idx]
        self.latent_vector=np.matmul(self.components_mean,self.transformation_mat)
        self.set_pix_map()

    def reset_play_speed(self):
        self.speed.setValue(-30)
        self.play_speed=30

    def set_speed(self):
        self.play_speed = -self.speed.value()

    def note_scale(self):
        self.s=self.scale.value()
        self.set_pix_map(scaled=True)
    def play_music(self):
        array_to_midi(np.transpose(self.music,axes=(1,0))*128,"./generated_music/foo",tempo=self.play_speed,threshold = self.scale.value(),dur = self.note_duration)
        
        pygame.mixer.music.stop()
        pygame.mixer.music.load("./generated_music/foo.mid")
        pygame.mixer.music.play()
    def reset_values(self):
        for i in range(2*self.num_sliders):
            # self.main_sliders[i].setValue(self.default_value[i]*self.precision)
            self.main_sliders[i].setValue(0)
            # self.fine_tune_sliders[i].setValue(0)
        self.scale.setValue(60)
    def stop_music(self):
        pygame.mixer.music.stop()

    def find_nearest(self):
        if not self.loaded:
            print("data not loaded, start loading now")
            time_step = 512
            self.population = load_all()
            # self.population = preprocess(data)
            batch_size = 64
            sample_batches = self.population.shape[0]//batch_size
            sample_mu = []
            for i in range(sample_batches):
                print("sample:", i,end="\r")
                x = self.population[i*batch_size:(i+1)*batch_size]
                z_mu = self.model.get_mu(x)
                z_mu=z_mu.numpy()
                sample_mu.append(z_mu)
            for i in range(sample_batches*batch_size,self.population.shape[0]):
                print("sample:", i,end="\r")
                x = self.population[i]
                z_mu=self.model.get_mu(np.expand_dims(x,axis=0))
                z_mu=z_mu.numpy()
                sample_mu.append(z_mu)
            print("done")
            self.population_mu = np.concatenate(sample_mu)
        self.loaded = True
        print("data loaded, start finding nearest neighbor")
        dist = np.sqrt(np.sum((self.latent_vector-self.population_mu)**2, axis=1))
        nn_idx = np.argsort(dist)[:self.k]
        for idx,i in enumerate(nn_idx):
            array_to_midi(np.transpose(self.population[i]*128,axes=(1,0)),"./nn/"+str(idx),tempo=self.play_speed)
        print("found, midi pieces under ./nn/")

        

app = qa(sys.argv)
model = Model()



load = 1

ckpt = tf.train.Checkpoint(model)
path = os.path.join(os.path.dirname(__file__),"saved_model/save_mode/model")
if load:
    ckpt.read(path)

# mixer config
freq = 44100  # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2  # 1 is mono, 2 is stereo
buffer = 1024   # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)

# optional volume 0 to 1.0
pygame.mixer.music.set_volume(0.8)

mixer = Mixer(200,200,1500,900,model)
sys.exit(app.exec_())