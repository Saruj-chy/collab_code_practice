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

    |	image_path |	target |
0   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
1   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
2   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
3   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
4   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
5   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
6   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
7   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
8   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	
9   |  	./drive/MyDrive/Snake_Dataset_All_In_One_Folde...   |  		walls_krait   |  	







<!-- 


```

```


```

```


```

``` -->