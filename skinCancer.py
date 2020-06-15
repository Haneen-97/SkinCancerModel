import os
import keras.backend as K
import numpy as np
import pandas as pd
from keras import Sequential
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import Flatten, Conv2D, Dense, MaxPooling2D, Dropout
from keras.utils import to_categorical

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


ImageClassMapping_path = "HAM10000_metadata.csv"
ClassLabels_path = "ClassLabels.xlsx"
ImagesRoot_path = "base_dir/"

df_ImageClassPath = pd.read_csv(ImageClassMapping_path)
# load Class Labels
df_Classes = pd.read_excel(ClassLabels_path)
df_ImageClassPath.groupby("ClassId").size().describe()
ddata = {"samples destribution": df_ImageClassPath.groupby("ClassId").size()}
iindex = range(2)

ddataframe = pd.DataFrame(data=ddata, index=iindex)


def SplitData(predictions, testsize):
    min = df_ImageClassPath.groupby("ClassId").size().min()

    df_TrainingSet = df_ImageClassPath[0:0].copy()
    df_TestSet = df_ImageClassPath[0:0].copy()
    df_PredSet = df_ImageClassPath[0:0].copy()

    for index, row in df_Classes.iterrows():
        df_FullSet = df_ImageClassPath[df_ImageClassPath['ClassId'] == row['ClassId']].sample(min, random_state=42)


        df_PredSet = df_PredSet.append(df_FullSet.sample(n=predictions, random_state=1))
        df_FullSet = pd.merge(df_FullSet, df_PredSet, indicator=True,
                              how='left').query('_merge=="left_only"').drop('_merge', axis=1)

        trainingSet, testSet = train_test_split(df_FullSet, test_size=testsize)
        df_TrainingSet = df_TrainingSet.append(trainingSet)
        df_TestSet = df_TestSet.append(testSet)

    return df_TrainingSet, df_TestSet, df_PredSet


def getDataSet(setType, isDL):
    imgs = []
    lbls = []
    df = pd.DataFrame(None)

    if setType == 'Training':
        df = dtTraining.copy()
    elif setType == 'Test':
        df = dtTest.copy()
    elif setType == 'Prediction':
        df = dtPred.copy()

    for index, row in df.iterrows():
        lbls.append(row['ClassId'])
        try:
            imageFilePath = os.path.join(ImagesRoot_path, row['x'])
            img = image.load_img(imageFilePath, target_size=(224, 224, 3),
                                 color_mode="rgb")
            img = image.img_to_array(img)  # to array
            img = img / 255  # Normalize
            imgs.append(img)

        except Exception as e:
            print(e)

    shuffle(imgs, lbls, random_state=255)  # Shuffle the dataset

    imgs = np.array(imgs)
    lbls = np.array(lbls)
    if isDL == True:
        lbls = to_categorical(lbls) 
    return imgs, lbls


def get_classlabel(class_code):
    return df_Classes.loc[df_Classes['ClassId'] == class_code, 'Class'].values[0]

def display_prediction(col_size, row_size, XPred, yPred):
    img_index = 0
    fig, ax = matplotlib.pyplot.subplots(row_size, col_size, figsize=(row_size * 4.5, col_size * 3.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(XPred[img_index][:, :, 0])
            print("h")
            ax[row][col].set_title("({}) {}".format(yPred[img_index], get_classlabel(yPred[img_index])))
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
            img_index += 1
    fig.tight_layout(h_pad=5, w_pad=5)

dtTraining, dtTest, dtPred = SplitData(5, 0.3)

ddata = {"Training": dtTraining.groupby("ClassId").size(), "Test": dtTest.groupby("ClassId").size()}
iindex = range(2)

ddataframe = pd.DataFrame(data=ddata, index=iindex)
X_train, y_train = getDataSet('Training', True)
X_valid, y_valid = getDataSet('Test', True)
X_pred, _ = getDataSet('Prediction', True)
print("Shape of Train Images:{} , Train Labels: {}".format(X_train.shape, y_train.shape))

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(96, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', sensitivity, specificity])

model.summary()
trained = model.fit(X_train, y_train, epochs=35, validation_data=(X_valid, y_valid))
model.save("SkinCancer_CNN.model")

plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(trained.history['sensitivity'])
plt.plot(trained.history['val_sensitivity'])
plt.title('Model sensitivity')
plt.ylabel('sensitivity')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(trained.history['specificity'])
plt.plot(trained.history['val_specificity'])
plt.title('Model specificity')
plt.ylabel('specificity')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

cnn_Y_pred = model.predict(X_valid)
cnn_Y_pred = np.argmax(cnn_Y_pred, axis=1)

# to display prediction images in subplots
display_prediction(5,2,X_pred,cnn_Y_pred)

