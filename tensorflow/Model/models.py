import tensorflow as tf

class ModelBase(tf.keras.Model):
    def preprocess(self,):
        pass
    
    def predict(self):
        pass
    
    
    
class UNet_v1(ModelBase):
    def __init__(self):
        pass
    
    def call(self,batch_images):
        pass
        
    def batch_predict(self,batch_image):
        pass
    
    def predict(self,image):
        pass