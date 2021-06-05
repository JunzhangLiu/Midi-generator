import numpy as np
import pretty_midi
import tensorflow as tf
import tensorflow.keras as keras
import random
from PIL import Image
from midi import *
from model import Model
from hyper_param import *
from pca import save_params

def save_sample(itr,data,validate=True):
    if validate:
        song = random.randrange(0,data.shape[0])
        seq = data[song]
        seq = model(np.expand_dims(seq,axis=0),training=False)
        seq = seq.numpy()
        seq = np.squeeze(seq,axis=0)
        seq*=128
        array_to_midi(np.transpose(seq,axes=(1,0)),"testing_data/"+str(itr)+"fit",tempo=5)
    num_song = 3
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
# opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer="adam")
load = 0

ckpt = tf.train.Checkpoint(model)
path = os.path.join(os.path.dirname(__file__),"saved_model/save_mode/model")
# path_b = os.path.join(os.path.dirname(__file__),"saved_model/backup_model/model")
if load:
    ckpt.read(path)

# for i in range(TRAIN_STEPS):
for i in range(TRAIN_STEPS):
    print(i)
    model.fit(x=data,batch_size=BATCH_SIZE,epochs=EPOCHS,shuffle=True)
    # if i % 2 == 0:
    np.random.shuffle(data)
    save_sample(i,data)
    # if i % 2 == 0 and i != 0:
    ckpt.write(path)
    # ckpt.write(path_b)
save_params(data, model)

