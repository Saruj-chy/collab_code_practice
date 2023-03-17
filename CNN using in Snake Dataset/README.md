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

<img src="https://raw.githubusercontent.com/Saruj-chy/collab_code_practice/main/CNN%20using%20in%20Snake%20Dataset/sample1.PNG" alt="Smaples of images" title="Optional title">
<img src="https://raw.githubusercontent.com/Saruj-chy/collab_code_practice/main/CNN%20using%20in%20Snake%20Dataset/sample2.PNG" alt="Smaples of Images" title="Optional title">

```
def show_model_history(modelHistory, model_name):
    history=pd.DataFrame()
    history["Train Loss"]=modelHistory.history['loss']
    history["Validatin Loss"]=modelHistory.history['val_loss']
    history["Train Accuracy"]=modelHistory.history['accuracy']
    history["Validatin Accuracy"]=modelHistory.history['val_accuracy']
  
    history.plot(figsize=(12,8))
    plt.title(" Convulutional Model {} Train and Validation Loss and Accuracy History".format(model_name))
    plt.show()
```

```
model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(WIDTH, HEIGHT, 3)))
model.add(layers.Conv2D(32, (3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), activation="relu"))
model.add(layers.Conv2D(128, (3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()
```

# Output:
```

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896       
                                                                 
 conv2d_1 (Conv2D)           (None, 146, 146, 32)      9248      
                                                                 
 batch_normalization (BatchN  (None, 146, 146, 32)     128       
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 73, 73, 32)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 73, 73, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 71, 71, 64)        18496     
                                                                 
 conv2d_3 (Conv2D)           (None, 69, 69, 64)        36928     
                                                                 
 batch_normalization_1 (Batc  (None, 69, 69, 64)       256       
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 34, 34, 64)       0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 34, 34, 64)        0         
                                                                 
 conv2d_4 (Conv2D)           (None, 32, 32, 128)       73856     
                                                                 
 conv2d_5 (Conv2D)           (None, 30, 30, 128)       147584    
                                                                 
 batch_normalization_2 (Batc  (None, 30, 30, 128)      512       
 hNormalization)                                                 
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 15, 15, 128)      0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 15, 15, 128)       0         
                                                                 
 conv2d_6 (Conv2D)           (None, 13, 13, 64)        73792     
                                                                 
 conv2d_7 (Conv2D)           (None, 11, 11, 64)        36928     
                                                                 
 batch_normalization_3 (Batc  (None, 11, 11, 64)       256       
 hNormalization)                                                 
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 5, 5, 64)          0         
                                                                 
 flatten (Flatten)           (None, 1600)              0         
                                                                 
 dense (Dense)               (None, 512)               819712    
                                                                 
 dropout_4 (Dropout)         (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 1,219,105
Trainable params: 1,218,529
Non-trainable params: 576
_________________________________________________________________

```

```
model.compile(loss="binary_crossentropy", 
             optimizer=optimizers.RMSprop(learning_rate=1e-4),
             metrics=["accuracy"])

```


```
dataset_train, dataset_test=train_test_split( 
    dataset,
    test_size=0.2,
    random_state=42
)

```


```
train_datagen=ImageDataGenerator(
rotation_range=15,
rescale=1./255,
shear_range=0.1,
zoom_range=0.2,
horizontal_flip=True,
width_shift_range=0.1,
height_shift_range=0.1)

train_datagenerator=train_datagen.flow_from_dataframe(dataframe=dataset_train,
                                                     x_col="image_path",
                                                     y_col="target",
                                                     target_size=(WIDTH, HEIGHT),
                                                    #  class_mode="binary",
                                                     batch_size=150)

```
Found 3647 validated image filenames belonging to 11 classes.


```
test_datagen=ImageDataGenerator(rescale=1./255)
test_datagenerator=test_datagen.flow_from_dataframe(dataframe=dataset_test,
                                                   x_col="image_path",
                                                   y_col="target",
                                                   target_size=(WIDTH, HEIGHT),
                                                  #  class_mode="binary",
                                                   batch_size=150)
```
Found 912 validated image filenames belonging to 11 classes.


```
modelHistory=model.fit(train_datagenerator,
                                epochs=5,
                                validation_data=test_datagenerator,
                                validation_steps=dataset_test.shape[0]//150,
                                steps_per_epoch=dataset_train.shape[0]//150
                                )


print("Train Accuracy:{:.3f}".format(modelHistory.history['accuracy'][-1]))
print("Test Accuracy:{:.3f}".format(modelHistory.history['val_accuracy'][-1]))
show_model_history(modelHistory=modelHistory, model_name="")
```

# Output

```
Epoch 1/5
24/24 [==============================] - 1339s 55s/step - loss: 0.6135 - accuracy: 0.8368 - val_loss: 0.9526 - val_accuracy: 0.0909
Epoch 2/5
24/24 [==============================] - 871s 36s/step - loss: 0.4835 - accuracy: 0.8679 - val_loss: 1.0676 - val_accuracy: 0.0909
Epoch 3/5
24/24 [==============================] - 878s 37s/step - loss: 0.4263 - accuracy: 0.8817 - val_loss: 1.0836 - val_accuracy: 0.0909
Epoch 4/5
24/24 [==============================] - 863s 36s/step - loss: 0.3851 - accuracy: 0.8922 - val_loss: 1.5573 - val_accuracy: 0.0909
Epoch 5/5
24/24 [==============================] - 865s 36s/step - loss: 0.3512 - accuracy: 0.9046 - val_loss: 2.3838 - val_accuracy: 0.0909
Train Accuracy:0.905
Test Accuracy:0.091

```


<img src="https://raw.githubusercontent.com/Saruj-chy/collab_code_practice/main/CNN%20using%20in%20Snake%20Dataset/graph_sample.PNG" alt="Convulational Model Train and Validation Loss and Accuracy History" title="Optional title">


```
model=applications.VGG16(weights="imagenet", include_top=False, input_shape=(WIDTH, HEIGHT, 3))
model.summary()
```

# Output
```
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
58889256/58889256 [==============================] - 0s 0us/step
Model: "vgg16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 150, 150, 3)]     0         
                                                                 
 block1_conv1 (Conv2D)       (None, 150, 150, 64)      1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 150, 150, 64)      36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 75, 75, 64)        0         
                                                                 
 block2_conv1 (Conv2D)       (None, 75, 75, 128)       73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 75, 75, 128)       147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 37, 37, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 37, 37, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 37, 37, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 37, 37, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 18, 18, 256)       0         
                                                                 
 block4_conv1 (Conv2D)       (None, 18, 18, 512)       1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 18, 18, 512)       2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 18, 18, 512)       2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 9, 9, 512)         0         
                                                                 
 block5_conv1 (Conv2D)       (None, 9, 9, 512)         2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 9, 9, 512)         2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 9, 9, 512)         2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 4, 4, 512)         0         
                                                                 
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
```

```
counter=0
features=list()
for path, target in zip(full_paths, targets):
    img=load_img(path, target_size=(WIDTH, HEIGHT))
    img=img_to_array(img)
    img=np.expand_dims(img, axis=0)
    feature=model.predict(img)
    features.append(feature)
    counter+=1
    if counter%2500==0:
        print("[INFO]:{} images loaded".format(counter))
```

```
features=np.array(features)
print("Before reshape,features.shape:",features.shape)
features=features.reshape(features.shape[0], 4*4*512)
print("After reshape, features.shape:",features.shape)
```
Before reshape,features.shape: (4559, 1, 4, 4, 512)
After reshape, features.shape: (4559, 8192)


```
le=LabelEncoder()
targets=le.fit_transform(targets)
print("features.shape:",features.shape)
print("targets.shape:",targets.shape)
```

features.shape: (4559, 8192)
targets.shape: (4559,)

```
X_train, X_test, y_train, y_test=train_test_split(features, targets, test_size=0.2, random_state=42)
```

```
from sklearn.model_selection import cross_val_score

clf=LogisticRegression(solver="lbfgs")
print("{} training...".format(clf.__class__.__name__))
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("The model trained and used to predict the test data...")
```

```
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:\n",metrics.classification_report(y_test, y_pred, target_names=["branded_krait", "binoellate_cobra", 
                                                                                             "common_krait","greater_black_krait", 
                                                                                             "green_pit", "king_cobra", 
                                                                                             "lesser_black_krait","monocellate_cobra", 
                                                                                             "russels_viper", "red_tailed_pit", 
                                                                                             "walls_krait"]))

```


# Output
```
Accuracy: 0.6743421052631579
Confusion Matrix:
 [[ 40   1   2   2   0   2   0   0   0   2   1]
 [  1  81   0   1   2   8   2  19   0   7   1]
 [  0   2 123   0   4   0   2   0   0   2  14]
 [  1   0   0  29   0   0   0   0  36   0   0]
 [  0   0   5   1  15   0   8   2   0   0   2]
 [  1  10   0   0   0  18   1  11   0   3   5]
 [  0   1   1   0   5   0  12   0   0   1   2]
 [  0  20   1   0   2   7   1  64   1   2   4]
 [  0   0   0  23   0   0   0   0  39   0   0]
 [  0   5   5   1   0   2   0   6   0 166   1]
 [  0   5  26   0   2   1   1   1   1   8  28]]
Classification Report:
                      precision    recall  f1-score   support

      branded_krait       0.93      0.80      0.86        50
   binoellate_cobra       0.65      0.66      0.66       122
       common_krait       0.75      0.84      0.79       147
greater_black_krait       0.51      0.44      0.47        66
          green_pit       0.50      0.45      0.48        33
         king_cobra       0.47      0.37      0.41        49
 lesser_black_krait       0.44      0.55      0.49        22
  monocellate_cobra       0.62      0.63      0.62       102
      russels_viper       0.51      0.63      0.56        62
     red_tailed_pit       0.87      0.89      0.88       186
        walls_krait       0.48      0.38      0.43        73

           accuracy                           0.67       912
          macro avg       0.61      0.60      0.60       912
       weighted avg       0.67      0.67      0.67       912

```






