import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Flatten, ReLU
from tensorflow.keras.models import Model

    
keras_layer = hub.KerasLayer('https://kaggle.com/models/spsayakpaul/vision-transformer/frameworks/TensorFlow2/variations/vit-l16-classification/versions/1')

"""local model"""
def VisionTransformer(input_shape = [224, 224]):
    inputs = Input(shape = (input_shape[0], input_shape[1], 3))
    vit16 = keras_layer(inputs)
    x = Dense(class_count, activation="softmax")(vit16)
    
    return Model(inputs, x)
    
    
"""global model"""
def global_model(img_size):
    img_shape = (img_size[0], img_size[1], 3)
    base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=img_shape, pooling='max')
    
    base_model.trainable = False
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = ReLU()(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006), 
              bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
    x = Dropout(rate=.4, seed=123)(x)
    output = Dense(class_count, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    return model
    
    
def fusion_model(local_model, global_model):
    shared_input = Input(shape=(224, 224, 3))
    local_output = local_model(shared_input)
    global_output = global_model(shared_input)
    fusion_layer = keras.layers.concatenate([local_output, global_output], axis=-1)
    fused_dense = Dense(16, activation='relu')(fusion_layer)
    output = Dense(class_count, activation='softmax')(fused_dense)
    
    model = Model(shared_input, output)
    
    return model
