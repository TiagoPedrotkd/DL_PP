# Data Preprocessing 

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from google.colab import drive
drive.mount("/content/drive")

dataset = pd.read_csv('/content/drive/My Drive/Churn.csv')
X = dataset.iloc[:, 3:13].values
# y is a vector containing the information about the target
y = dataset.iloc[:, 13].values

# categorical data to numerical data
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(drop="first"), [1,2])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Creation of training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Standardization only uses the information on the training set!!!
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Creation of the model
import keras
from keras.models import Sequential
from keras.layers import Dense

# Creation of an empty model
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Training the network 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
hist=classifier.fit(X_train, y_train, validation_split= 0.3, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Print Statistics from Confusion Matrix
from sklearn.metrics import classification_report
target_names = ['Stay', 'Leave']
print(classification_report(y_test, y_pred, target_names=target_names))

import seaborn as sns
import matplotlib.pyplot as plt
# Visualizing Confusion Matrix
plt.figure(figsize = (8, 5))
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15}, 
            yticklabels = ['Stay', 'Leave'], xticklabels = ['Predicted Stay', 'Predicted Leave'])
plt.yticks(rotation = 0)
plt.show()



loss_train = hist.history['accuracy']
loss_val = hist.history['val_accuracy']
epochs = range(1,101)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss_train = hist.history['loss']
loss_val = hist.history['val_loss']
epochs = range(1,101)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()