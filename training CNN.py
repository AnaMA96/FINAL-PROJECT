import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import models
from keras import layers
from sklearn import preprocessing 
import json
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import ast


df_train = pd.read_pickle("./transformed_2.csv")
df_train.shape

X = np.vstack(df_train["transformed"])
y = df_train.type

print(y)

label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(df_train.type)


df_train.numeric = label_encoded
df_train.groupby(by="numeric").head()

label_encoded = label_encoded[:, np.newaxis]
print(label_encoded)

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(label_encoded)
print(one_hot_encoded)

X = np.vstack(df_train.transformed)
y = one_hot_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


number_classes = 8
inshape = (X_train.shape[1],)


model = models.Sequential()  #creates the neural network layers:

model.add(layers.Dense(512, activation='relu', input_shape=inshape))
model.add(layers.Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(number_classes, activation='softmax'))
model.compile(optimizer='Nadam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

history = model.fit(X_train,
        y_train,
        epochs=100,
        batch_size=20,
        validation_data=(X_test, y_test))

results = model.evaluate(X_test, y_test)
print("\n")
print("Resultados: ",results)
print("\n")

name="models/0.9829-accuracy-200000-50epochs-loss0.0823"


model_json = model.to_json()
with open(name+'.json', "w") as json_file:
    json.dump(model_json, json_file)
model.save(name+'.h5')

# Displaying loss values

plt.figure(figsize=(8,8))
plt.title('Loss Value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
print('loss:', history.history['loss'][-1])
print('val_loss:', history.history['val_loss'][-1])
plt.show()

# Displaying accuracy scores
plt.figure(figsize=(8,8))
plt.title('Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
print('accuracy:', history.history['accuracy'][-1])
print('val_accuracy:', history.history['val_accuracy'][-1])
plt.show()

#Model evaluation

predictions = model.predict(X_test)
predictions

'''The code below shows how to take the argmax of all predictions on X_test and 
then followed by decoding y_test into the same form as the predictions variable 
(because previously we already converted y_test into one-hot representation, now we need to convert 
 that back to label-encoded form). This is extremely necessary to do because we want to compare each of 
the element of predictions and y_test.'''

predictions = np.argmax(predictions, axis=1)
y_test = one_hot_encoder.inverse_transform(y_test)

# Creating confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()