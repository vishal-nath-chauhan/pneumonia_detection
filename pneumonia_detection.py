
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import cv2 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from skimage import color,transform,io
import os
import skimage
tf.get_logger().setLevel('INFO')


class disease:
    def __init__(self,model_path):
        # self.img_path=img_path
        self.model_path=model_path
        self.image_size=224
        self.labels=['NORMAL', 'PNEUMONIA'] 
        self.K=K
        self.np=np
        self.cv2=cv2
        self.keras=keras
        self.tf=tf

    def f1(self,y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = self.K.sum(self.K.round(self.K.clip(y_true * y_pred, 0, 1)))
            possible_positives = self.K.sum(self.K.round(self.K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + self.K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = self.K.sum(self.K.round(self.K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = self.K.sum(self.K.round(self.K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + self.K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+self.K.epsilon()))


    


    def predict(self,image_path):
        dependencies = {
        'f1': self.f1  
        }
        with self.tf.device('/CPU:0'):
            mo=self.keras.models.load_model(self.model_path,custom_objects=dependencies)
        img=self.cv2.imread(image_path)
        image_resized = self.cv2.resize(img,(self.image_size,self.image_size))
        imgg=self.cv2.cvtColor(image_resized, self.cv2.COLOR_BGR2RGB)
        dt=self.np.array(imgg)
        dt=dt.reshape((1,224,224,3))
        pred=mo.predict(dt)
        prediction=pred[0][0]
        return prediction
        # if prediction==1:     
        #     return 1
        # if prediction==0:
        #     return 0
# 1==> Pneumonia
# 0==>No pneumonia
# print(predict('/workspace/django-locallibrary-tutorial/pneumonia_detection/test_images/normal.jpeg'))
