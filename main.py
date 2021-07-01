# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# import dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# use scikit learn
from sklearn.model_selection import train_test_split

# initialize ANN as sequence of layers as in ANN architecture
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))

# compile the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# train the ANN model
ann.fit(X_train, y_train, batch_size=32, epochs = 100)

# Predicting the results of the Test set
y_pred = ann.predict(x=X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))
