"""
This code is for global model VGG16 and
is the visualization of class activation map using Grad-CAM algorithm
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from matplotlib import cm


last_conv_layer_name = "block5_conv3"
classifier_layer_names = [
    "global_max_pooling2d_1",
    "batch_normalization_1",
    "re_lu_1",
    "dense_2",
    "dropout_1",
    "dense_3"
]


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap
  
  
heatmap = make_gradcam_heatmap(
    x, model, last_conv_layer_name, classifier_layer_names
)

img = image.load_img(img_path)
x = image.img_to_array(img)
 
heatmap = np.uint8(255 * heatmap)
 
# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")
 
# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]
 
# We create an image with RGB colorized heatmap
jet_heatmap = image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((x.shape[1], x.shape[0]))
jet_heatmap = image.img_to_array(jet_heatmap)
 
# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + x
superimposed_img = image.array_to_img(superimposed_img)

save_path = "cam.jpg"
superimposed_img.save(save_path)
