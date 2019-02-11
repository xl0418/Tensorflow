# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# print(tf.__version__)
# Load the data to train
file='C:/Liang/Code/Pro2/tf_classification/'
# # file = '/home/p274981/abcpp/abcpp/'

mode = ['BM','OU','TP','antiOU']
for count in range(len(mode)):
    model = mode[count]
    file = 'C:/Liang/Code/Pro2/tf_classification/%s' % model
    # # file = '/home/p274981/abcpp/abcpp/'
    filename = file + 'tf_tree2modelseletrain.npy'
    data_train = np.load(filename).item()
    filename_test = file + 'tf_tree2modelseletest.npy'
    data_test = np.load(filename_test).item()
    if count == 0:
        train_Z_model = data_train['Z_train']
        train_label_model = data_train['Z_labels']
        test_Z_model = data_test['Z_train']
        test_label_model = data_test['Z_labels']
    else:
        train_Z_model = np.concatenate((train_Z_model,data_train['Z_train']))
        train_label_model = np.concatenate((train_label_model,data_train['Z_labels']))
        test_Z_model = np.concatenate((test_Z_model,data_test['Z_train']))
        test_label_model = np.concatenate((test_label_model,data_test['Z_labels']))

#
# # Load the data to test
# file='C:/Liang/Code/Pro2/tf_classification/'
# # # file = '/home/p274981/abcpp/abcpp/'
# filename_test = file + 'tf_tree2modelseletest.npy'
# data_test = np.load(filename_test).item()


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


# Normalize the test data
Z_test = test_Z_model
Z_test_normalized = (Z_test-np.min(Z_test))/(np.max(Z_test)-np.min(Z_test))

Labels_test = test_label_model

predictions = model.predict(Z_test_normalized)
predictions[1]
for i in range(len(predictions)):
    print(np.argmax(predictions[i])- Labels_test[i])


xticks = [0,1,2,3]
xticklabels = ['B','O','T','A']
num_rows = 19
num_cols = 10
num_images = num_rows*num_cols
fig,axes = plt.subplots(num_rows,num_cols,figsize = (2*num_cols, num_rows),sharex=True,sharey=True)
count_fig = 0
for row in range(num_rows):
    for col in range(num_cols):

        # axes[i].grid(False)
        # axes[i].set_xticks([])
        # axes[i].set_yticks([])
        predicted_label = np.argmax(predictions[count_fig])
        axes[row,col].bar(range(classnum), predictions[count_fig], color="grey")
        axes[row,col].set_ylim([0, 1])
        axes[row,col].set_yticks([])
        axes[row,col].set_yticklabels([])
        axes[row,col].set_xticks(xticks)
        axes[row,col].set_xticklabels(xticklabels)

        axes[row,col].get_children()[predicted_label].set_color('red')
        axes[row,col].get_children()[int(Labels_test[count_fig])].set_color('blue')
        count_fig +=1

fig.text(0.5, 0.04, 'Models', ha='center', fontsize=15)
fig.text(0.04, 0.5, 'Probability', va='center', rotation='vertical', fontsize=15)
fig.suptitle('Model classification via machine learning')


# single plot of prediction
num = 190
Labels_test = test_label_model

predicted_label = np.argmax(predictions[num])
barplot = plt.bar(range(classnum), predictions[num],color = 'grey')
barplot.get_children()[predicted_label].set_color('red')
barplot.get_children()[int(Labels_test[num])].set_color('blue')
plt.xticks(xticks,xticklabels)
plt.xlabel("Models")
plt.ylabel("Probability")
plt.title("Prediction")
