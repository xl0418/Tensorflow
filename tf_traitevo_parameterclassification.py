# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)
# Load the data to train
file='C:/Liang/Code/Pro2/tf_classification/'
# # file = '/home/p274981/abcpp/abcpp/'
filename = file + 'tf_trait2.npy'
data_train = np.load(filename).item()

# Load the data to test
file='C:/Liang/Code/Pro2/tf_classification/'
# # file = '/home/p274981/abcpp/abcpp/'
filename_test = file + 'tf_c1_test.npy'
data_test = np.load(filename_test).item()


# Normalize the trait data
Z_train = data_train['Z_train']
Z_train_normalized = (Z_train-np.min(Z_train))/(np.max(Z_train)-np.min(Z_train))

Labels_train = data_train['Z_labels']
classnum = len(np.unique(Labels_train))
#
model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(classnum, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(Z_train_normalized, Labels_train, epochs=10)

test_loss, test_acc = model.evaluate(Z_train_normalized, Labels_train)

print('Test accuracy:', test_acc)

# predict for a single image

# Normalize the test data
Z_test = data_test['Z_train']
Z_test_normalized = (Z_test-np.min(Z_test))/(np.max(Z_test)-np.min(Z_test))

Labels_test = data_test['Z_labels']

predictions = model.predict(Z_test_normalized)
predictions[1]
for i in range(len(predictions)):
    print(np.argmax(predictions[i])- Labels_test[i])



def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(6), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[int(true_label)].set_color('blue')

num_rows = 7
num_cols = 7
num_images = num_rows*num_cols
plt.figure(figsize=(2*num_cols, num_rows))
for i in range(num_images):
  plt.subplot(num_rows, num_cols, i+1)
  plot_value_array(i, predictions, Labels_test)
