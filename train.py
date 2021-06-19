import numpy as np
import pretty_midi
import tensorflow as tf
import tensorflow.keras as keras
import random
from PIL import Image
from midi import *
from model import Model
from hyper_param import *
from pca import save_components

def save_sample(itr,data,num_song=4):
    for i in range(num_song):
        z = np.random.normal(0.0,1,(1,LATENT_DIM))
        seq = model.generate_from_latent(z)
        seq=seq.numpy()
        seq=np.squeeze(seq,axis=0)
        
        seq*=128
        array_to_midi(np.transpose(seq,axes=(1,0)),"testing_data/"+str(itr)+"_rand"+str(i),tempo=5,threshold=0.25*128)


data = load_all()
print(data.shape)
model = Model()
model.compile(optimizer="adam",loss="binary_crossentropy")
load = 0
ckpt = tf.train.Checkpoint(model)
path = os.path.join(os.path.dirname(__file__),"saved_model/save_model/model")
if load:
    ckpt.read(path)

np.random.seed(0)  
np.random.shuffle(data)#ensure the data is always shuffled the same way
train_set_sz = int(data.shape[0]/BATCH_SIZE)*BATCH_SIZE
data=data[:train_set_sz] #ensure the training data size is a multiple of batch size, otherwise LSTM might fail
print(data.shape)
for i in range(TRAIN_STEPS):
    print(i)
    model.fit(x=data,batch_size=BATCH_SIZE,epochs=EPOCHS_PER_STEP,shuffle=True)
    if i > START_SAVE:
        ckpt.write(os.path.join(os.path.dirname(__file__),"saved_model/model"+str(i)+"/model"))
save_components(data, model)

