import os
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle # Shuffle arrays or sparse matrices in a consistent way
from sklearn.model_selection import train_test_split, \
    GridSearchCV
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, MaxPooling2D, Dropout
from keras.utils import to_categorical
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasClassifier


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
ImageClassMapping_path = "HAM10000_metadata.csv"
ClassLabels_path = "ClassLabels.xlsx"
ImagesRoot_path = "base_dir/"

ModelFileName ='SkinCancer_CNN.model'
df_ImageClassPath = pd.read_csv(ImageClassMapping_path)
# load Class Labels
df_Classes = pd.read_excel(ClassLabels_path)
df_ImageClassPath.groupby("ClassId").size().describe()
ddata = {"samples destribution":df_ImageClassPath.groupby("ClassId").size()}
iindex = range(2)

ddataframe = pd.DataFrame(data=ddata, index= iindex)


def SplitData(predictions, testsize):
    min =df_ImageClassPath.groupby("ClassId").size().min()

    # empty dataframes with same column difinition
    df_TrainingSet = df_ImageClassPath[0:0].copy()
    df_TestSet = df_ImageClassPath[0:0].copy()
    df_PredSet = df_ImageClassPath[0:0].copy()

    # Create the sets by loop thru classes and append
    for index, row in df_Classes.iterrows():
        # make sure all class are same size
        df_FullSet = df_ImageClassPath[df_ImageClassPath['ClassId'] == row['ClassId']].sample(min, random_state=42)

        #         df_FullSet = df_ImageClassPath[df_ImageClassPath['ClassId'] == row['ClassId']]

        df_PredSet = df_PredSet.append(df_FullSet.sample(n=predictions, random_state=1))
        df_FullSet = pd.merge(df_FullSet, df_PredSet, indicator=True,
                              how='left').query('_merge=="left_only"').drop('_merge', axis=1)

        trainingSet, testSet = train_test_split(df_FullSet, test_size=testsize)
        df_TrainingSet = df_TrainingSet.append(trainingSet)
        df_TestSet = df_TestSet.append(testSet)

    return df_TrainingSet, df_TestSet, df_PredSet




def getDataSet(setType, isDL):  # 'Training' for Training dataset , 'Testing' for Testing data set
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
            if isDL == False:
                img = img.flatten()  # for knn_classifier Model
            imgs.append(img)

        except Exception as e:
            print(e)

    shuffle(imgs, lbls, random_state=255)  # Shuffle the dataset

    imgs = np.array(imgs)
    lbls = np.array(lbls)
    if isDL == True:
        lbls = to_categorical(lbls)  # for keras CNN Model
    return imgs, lbls
dtTraining, dtTest,dtPred = SplitData(3,0.3)


ddata = {"Training":dtTraining.groupby("ClassId").size(),"Test":dtTest.groupby("ClassId").size()}
iindex = range(2)
ddataframe = pd.DataFrame(data=ddata, index= iindex)
X_train,y_train = getDataSet('Training',True)
X_valid,y_valid = getDataSet('Test',True)
X_pred,_ = getDataSet('Prediction',True)
print("Shape of Train Images:{} , Train Labels: {}".format(X_train.shape,y_train.shape))




def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', sensitivity, specificity])

    return model

model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=128, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)


grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))