import numpy as np
from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense
data = np.random.random((1000,100))
labels = np.random.randint(2,size=(1000,1))
model = keras.Sequential()
model.add(keras.layers.Dense(32,
 activation='relu',
 input_dim=100))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
 loss='binary_crossentropy',
 metrics=['accuracy'])
model.fit(data,labels,epochs=10,batch_size=32)
loss_and_metrics = model.evaluate(data, labels, batch_size=128)
predictions = model.predict(data)