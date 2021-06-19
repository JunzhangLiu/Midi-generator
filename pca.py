import numpy as np
import pretty_midi
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.decomposition import PCA
from sklearn import preprocessing
from midi import *
from model import Model
from hyper_param import *
from saved_file_location import *
import matplotlib.pyplot as plt

def get_all_latent_vec(time_step,data,net,batch_size,bins):
    latent_vecs = []
    np.random.shuffle(data)
    sample_batches=data.shape[0]//batch_size
    for i in range(sample_batches):
        print("sample:", i,end="\r")
        x = data[i*batch_size:(i+1)*batch_size]
        x = tf.cast(x,tf.float16)
        latent_vec= net.get_latent_enc(x)
        latent_vec=latent_vec.numpy()
        latent_vecs.append(latent_vec)
        
    for i in range(sample_batches*batch_size,data.shape[0]):
        print("sample:", i,end="\r")
        x = data[i]
        x = tf.cast(x,tf.float16)
        latent_vec=net.get_latent_enc(np.expand_dims(x,axis=0))
        latent_vec=latent_vec.numpy()
        latent_vecs.append(latent_vec)
    for i in range(sample_batches*batch_size,data.shape[0]):
        print("sample:", i,end="\r")
        x = data[i]
        x = tf.cast(x,tf.float16)
        latent_vec=net.model.get_latent_enc(np.expand_dims(x,axis=0))
        latent_vec=latent_vec.numpy()
        latent_vecs.append(latent_vec)
    latent_vecs = np.concatenate(latent_vecs)
    return latent_vecs

def save_components(data,model):

    np.random.seed(0)  
    np.random.shuffle(data)#ensure the data is always shuffled the same way
    train_set_sz = int(data.shape[0]/BATCH_SIZE)*BATCH_SIZE
    data=data[:train_set_sz]
    
    latent_vecs = get_all_latent_vec(TIME_STEP,data,model,BATCH_SIZE,50)
    mean = np.mean(latent_vecs, axis=0)
    std = np.std(latent_vecs, axis=0)
    latent_vecs-=mean
    latent_vecs/=std

    pca = PCA()
    pca.fit(latent_vecs)
    pca_data=pca.transform(latent_vecs)
    print(pca_data.shape)
    std = np.std(np.matmul(latent_vecs,pca.components_.T),axis=0)
    np.save(COMPONENTS_MEAN_SAVE_LOCATION, np.mean(np.matmul(latent_vecs,pca.components_.T),axis=0))
    np.save(COMPONENTS_STD_SAVE_LOCATION,np.std(np.matmul(latent_vecs,pca.components_.T),axis=0))
    
    np.save(COMPONENTS_SAVE_LOCATION,np.linalg.inv(pca.components_).T)

if __name__ == '__main__':
    data = load_all()
    np.random.seed(0)  
    np.random.shuffle(data)
    train_set_sz = int(data.shape[0]/BATCH_SIZE)*BATCH_SIZE
    data=data[:train_set_sz] 
    model = Model()
    # path = os.path.join(os.path.dirname(__file__),"saved_model/save_mode/model")
    path = os.path.join(os.path.dirname(__file__),SAVED_MODEL_LOCATION)
    # path_b = os.path.join(os.path.dirname(__file__),"saved_model/backup_model/model")
    ckpt = tf.train.Checkpoint(model)
    ckpt.read(path)

    save_components(data,model)