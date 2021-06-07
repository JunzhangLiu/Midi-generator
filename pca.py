import numpy as np
import pretty_midi
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from midi import *
from model import Model
from hyper_param import *
from matplotlib import pyplot as plt

def get_popluation_mu(time_step,data,net,batch_size,sample_batches,bins):
    sample_mu = []
    np.random.shuffle(data)
    for i in range(sample_batches):
        print("sample:", i,end="\r")
        x = data[i*batch_size:(i+1)*batch_size]
        x = tf.cast(x,tf.float16)
        z_mu= net.get_mu(x)
        z_mu=z_mu.numpy()
        # z_sigma=z_sigma.numpy()
        sample_mu.append(z_mu)
        # sample_sigma.append(z_sigma)
        
    for i in range(sample_batches*batch_size,data.shape[0]):
        print("sample:", i,end="\r")
        x = data[i]
        x = tf.cast(x,tf.float16)
        z_mu=net.get_mu(np.expand_dims(x,axis=0))
        z_mu=z_mu.numpy()
        sample_mu.append(z_mu)
    sample_mu = np.concatenate(sample_mu)
    # sample_sigma = np.concatenate(sample_sigma)
    return sample_mu#,sample_sigma

def save_params(data,model):

    sample_batches = data.shape[0]//64

    num_song = 10
    sample_mu = get_popluation_mu(TIME_STEP,data,model,64,sample_batches,50)
    # d = preprocessing.scale(sample_mu)

    mean = np.mean(sample_mu, axis=0)
    std = np.std(sample_mu, axis=0)
    sample_mu-=mean
    sample_mu/=std

    pca = PCA()
    pca.fit(sample_mu)
    pca_data=pca.transform(sample_mu)
    print(pca_data.shape)
    std = np.std(np.matmul(sample_mu,pca.components_.T),axis=0)
    np.save(COMPONENTS_MEAN_SAVE_LOCATION, np.mean(np.matmul(sample_mu,pca.components_.T),axis=0))
    np.save(COMPONENTS_STD_SAVE_LOCATION,np.std(np.matmul(sample_mu,pca.components_.T),axis=0))
    
    np.save(COMPONENTS_SAVE_LOCATION,np.linalg.inv(pca.components_).T)
    # m = np.matmul(sample_mu,pca.components_.T)
    # plt.hist(m[:,0],bins=100,density=True)
    # plt.show()


if __name__ == '__main__':
    data = load_all()
    # data = preprocess(data)    
    model = Model()
    path = os.path.join(os.path.dirname(__file__),"saved_model/save_mode/model")
    # path_b = os.path.join(os.path.dirname(__file__),"saved_model/backup_model/model")
    ckpt = tf.train.Checkpoint(model)
    ckpt.read(path)

    save_params(data,model)