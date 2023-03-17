# Snake Image Identification Using CNN
This project is modified from [CAT & DOG Dataset](https://www.kaggle.com/code/serkanpeldek/keras-cnn-transfer-learnings-on-cats-dogs-dataset), which is used by CNN model in Keras.

### Quickstart
Mounted with google drive for accessing the data: 

```
from google.colab import drive
drive.mount('/content/drive')
```


### Import Library
```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import optimizers
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing.image import  ImageDataGenerator
from keras import applications


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import glob
import os
print("Snake Image Dataset Folder Contain:",os.listdir("./drive/MyDrive/Snake_Dataset_All_In_One_Folder"))
```

```
IMAGE_FOLDER_PATH="./drive/MyDrive/Snake_Dataset_All_In_One_Folder"
FILE_NAMES=os.listdir(IMAGE_FOLDER_PATH)
WIDTH=150
HEIGHT=150
```


```
targets=list()
full_paths=list()
for file_name in FILE_NAMES:
    target=file_name.split(".")[0]
    full_path=os.path.join(IMAGE_FOLDER_PATH, file_name)
    full_paths.append(full_path)
    targets.append(target)

dataset=pd.DataFrame()
dataset['image_path']=full_paths
dataset['target']=targets

dataset.head(10)
```

|   SL  |	            image_path                              |	    target        |
| ------| ----------------------------------------------------- | ------------------- |
|   0   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
|   1   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
|   2   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
|   3   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
|   4   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
|   5   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
|   6   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
|   7   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
|   8   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
|   9   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	



### Total Datasetin Drive
```
# branded_krait, binoellate_cobra , common_krait , greater_black_krait , green_pit , king_cobra , lesser_black_krait , 
# monocellate_cobra , russels_viper , red_tailed_pit , walls_krait

target_counts=dataset['target'].value_counts()
print("Number of branded_krait in the dataset:{}".format(target_counts['banded_krait']))
print("Number of binoellate_cobra in the dataset:{}".format(target_counts['binoellate_cobra']))
print("Number of common_krait in the dataset:{}".format(target_counts['common_krait']))
print("Number of greater_black_krait in the dataset:{}".format(target_counts['greater_black_krait']))
print("Number of green_pit in the dataset:{}".format(target_counts['grean_pit']))
print("Number of king_cobra in the dataset:{}".format(target_counts['king_cobra']))
print("Number of lesser_black_krait in the dataset:{}".format(target_counts['lesser_black_krait']))
print("Number of monocellate_cobra in the dataset:{}".format(target_counts['monocellate_cobra']))
print("Number of russels_viper in the dataset:{}".format(target_counts['russels_viper']))
print("Number of red_tailed_pit in the dataset:{}".format(target_counts['red_tailed_pit']))
print("Number of walls_krait in the dataset:{}".format(target_counts['walls_krait']))
```

Number of branded_krait in the dataset:279
Number of binoellate_cobra in the dataset:685
Number of common_krait in the dataset:691
Number of greater_black_krait in the dataset:182
Number of green_pit in the dataset:350
Number of king_cobra in the dataset:261
Number of lesser_black_krait in the dataset:97
Number of monocellate_cobra in the dataset:497
Number of russels_viper in the dataset:894
Number of red_tailed_pit in the dataset:290
Number of walls_krait in the dataset:333


```
def get_side(img, side_type, side_size=5):
    height, width, channel=img.shape
    if side_type=="horizontal":
        return np.ones((height,side_size,  channel), dtype=np.float32)*255
        
    return np.ones((side_size, width,  channel), dtype=np.float32)*255

def show_gallery(show="both"):
    n=100
    counter=0
    images=list()
    vertical_images=[]
    rng_state = np.random.get_state()
    np.random.shuffle(full_paths)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)
    for path, target in zip(full_paths, targets):
        if target!=show and show!="both":
            continue
        counter=counter+1
        if counter%11==0:
            break
        #Image loading from disk as JpegImageFile file format
        img=load_img(path, target_size=(WIDTH,HEIGHT))
        #Converting JpegImageFile to numpy array
        img=img_to_array(img)
        
        hside=get_side(img, side_type="horizontal")
        images.append(img)
        images.append(hside)

        if counter%10==0:
            himage=np.hstack((images))
            vside=get_side(himage, side_type="vertical")
            vertical_images.append(himage)
            vertical_images.append(vside)
            
            images=list()

    gallery=np.vstack((vertical_images)) 
    plt.figure(figsize=(12,12))
    plt.xticks([])
    plt.yticks([])
    title={"both":"Snake Dataset",
          "banded_krait": "banded_krait",
          "binoellate_cobra": "binoellate_cobra",
          "common_krait": "common_krait",
          "greater_black_krait": "greater_black_krait",
          "grean_pit": "grean_pit",
          "king_cobra": "king_cobra",
          "lesser_black_krait": "lesser_black_krait",
          "monocellate_cobra": "monocellate_cobra",
          "russels_viper": "russels_viper",
          "red_tailed_pit": "red_tailed_pit",
          "walls_krait": "walls_krait"}
    plt.title("10 samples of {} of the dataset".format(title[show]))
    plt.imshow(gallery.astype(np.uint8))
```

```
show_gallery(show="banded_krait")
show_gallery(show="binoellate_cobra")
show_gallery(show="common_krait")
show_gallery(show="greater_black_krait")
show_gallery(show="grean_pit")
show_gallery(show="king_cobra")
show_gallery(show="lesser_black_krait")
show_gallery(show="monocellate_cobra")
show_gallery(show="russels_viper")
show_gallery(show="red_tailed_pit")
show_gallery(show="walls_krait")
show_gallery(show="both")
```
















