import numpy as np
import os
from PIL import Image

class Helper():
    @staticmethod
    def load_data(train_name, label_name):
        imgs = os.listdir(train_name)
        num = len(imgs)

        data = np.empty((num - 1,3,64,64),dtype="float32")
        label = np.empty((num - 1,),dtype ="uint8")
        label = np.genfromtxt(label_name,delimiter=',')
        j=0
        for i in range(num):
            if not imgs[i].startswith('.'):
                img = Image.open(train_name+imgs[i])
                arr = np.asarray (img, dtype ="float32")
                data [j,:,:,:] = [arr[:,:,0],arr[:,:, 1],arr[:,:, 2]]
                j=j+1
        return data, label
