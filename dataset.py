import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


def make_dataframes(sdir):
    filepaths = []
    labels = []
    
    classlist = sorted(os.listdir(sdir)) # ['Benign Masses', 'Malignant Masses']
    
    for clas in classlist:
        classpath = os.path.join(sdir, clas)
        if os.path.isdir(classpath):
            img_name_list = sorted(os.listdir(classpath))
            desc = f"{clas:25s}"
            
            # progress bar
            for img_name in tqdm(img_name_list, ncols=130, desc=desc, unit='files',colour='blue'):
                img_path = os.path.join(classpath, img_name)
                filepaths.append(img_path)
                labels.append(clas)
    
    FileSeries = pd.Series(filepaths, name='filepaths')
    LabelSeries = pd.Series(labels, name='labels')
    
    df = pd.concat([FileSeries, LabelSeries], axis=1)
    
    train_df, dummy_df = train_test_split(df, train_size=.9, shuffle=True, random_state=123, stratify=df['labels'])
    valid_df, test_df = train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])
    
    classes = sorted(train_df['labels'].unique())
    class_count = len(classes)
    
    print('number of classes in processed dataset = ', class_count)
    print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df)) 
    
    return train_df, test_df, valid_df, classes, class_count
  
  
  
def make_gens(batch_size, train_df, test_df, valid_df, img_size):
    
    train_gen = ImageDataGenerator().flow_from_dataframe(train_df, 
                                                        x_col='filepaths', 
                                                        y_col='labels', 
                                                        target_size=img_size, 
                                                        class_mode='categorical', 
                                                        color_mode='rgb', 
                                                        shuffle=True, 
                                                        batch_size=batch_size)
    
    valid_gen = ImageDataGenerator().flow_from_dataframe(valid_df, 
                                                        x_col='filepaths', 
                                                        y_col='labels', 
                                                        target_size=img_size, 
                                                        class_mode='categorical', 
                                                        color_mode='rgb', 
                                                        shuffle=False, 
                                                        batch_size=batch_size)
    
    test_gen = ImageDataGenerator().flow_from_dataframe(test_df, 
                                                        x_col='filepaths', 
                                                        y_col='labels', 
                                                        target_size=img_size, 
                                                        class_mode='categorical', 
                                                        color_mode='rgb', 
                                                        shuffle=False, 
                                                        batch_size=batch_size)
    
    return train_gen, test_gen, valid_gen
  
  
def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)
    
    plt.figure(figsize=(25,25))
    length = len(labels)
    if length < 25:
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5,5,i+1)
        image = images[i] / 255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='red', fontsize=20)
        plt.axis('off')
    plt.show()
    
show_image_samples(train_gen)
