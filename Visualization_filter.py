"""
This code is run for global model VGG16
"""

import tensorflow as tf
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

# load a test image
img_path = '../input/breastcancermasses/Dataset of Mammography with Benign Malignant Breast Masses/Dataset of Mammography with Benign Malignant Breast Masses/INbreast+MIAS+DDSM Dataset/Malignant Masses/20587612 (37).png'

img = image.load_img(img_path)
img = img.resize((224, 224))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)
preds = model.predict(x)

layer_outputs = [layer.output for layer in model.layers[:12]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(x)

for i in range(len(activations)):
    print(activations[i].shape)

layer_names = []
for layer in model.layers[:6]:
    layer_names.append(layer.name)

# number of feature maps each row(columns)   
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
	# layer_activation: (1, size, size, channels)
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    
    # number of rows
    n_cols = n_features // images_per_row
    
    display_grid = np.zeros((size * n_cols, images_per_row*size))
    
    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image = layer_activation[0, :, :, col*images_per_row+row]
            
            """beautiful to look at"""
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0,  255).astype('uint8')
            
            display_grid[col*size:(col+1)*size,
                         row*size:(row+1)*size] = channel_image
    scale = 1./size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
