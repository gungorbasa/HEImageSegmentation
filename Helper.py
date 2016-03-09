import numpy as np
import os
from PIL import Image
from keras import models

class Helper():
    def _load_data(path):
        imgs = os.listdir(train_name)
        num = len(imgs)
        data = np.empty((num - 1,3,64,64),dtype="float32")
        j=0
        for i in range(num):
            if not imgs[i].startswith('.'):
                img = Image.open(train_name+imgs[i])
                arr = np.asarray (img, dtype ="float32")
                data [j,:,:,:] = [arr[:,:,0],arr[:,:, 1],arr[:,:, 2]]
                j=j+1
        return data

    @staticmethod
    def load_data(train_name,test_name, train_label_path, test_label_path):
        train_data = Helper._load_data(train_name)
        test_data = Helper._load_data(test_name)
        train_label = Helper.load_labels(train_label_path)
        test_label = Helper.load_labels(test_label_path)

        return train_data, train_label, test_data, test_label


    @staticmethod
    def load_labels(label_name):
        label = np.genfromtxt(label_name,delimiter=',')
        return label


    @staticmethod
    def save_model(model, name):
        json_string = model.to_json()
        open(name + '.json', 'w').write(json_string)
        model.save_weights(name + '_weights.h5')

    @staticmethod
    def load_model(name):
        model = models.model_from_json(open(name + '.json').read())
        model.load_weights(name + '_weights.h5')
        return model

    @staticmethod
    def pop_layers_from_model(model, n):
        poppedLayers = []
        for j in range(n):
            poppedLayers.append(model.layers.pop())

        return [model, poppedLayers]

    @staticmethod
    def append_layers(model, layers):
        n = len(layers)
        layers.reverse()
        for j in range(n):
            model.layers.append(layers[j])

        return model
