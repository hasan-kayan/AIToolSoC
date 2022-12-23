import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from time import perf_counter

def get_data(path):
    filenames_list = []
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file[-4:] == '.csv':
                filenames_list.append(os.path.join(root, file))
    return filenames_list

class Data_csv:
    def __init__(self, name, data):
        self.name = name
        self.data = data

data_list = get_data('Data')
class_data_list = []

i = 0
for i in range(len(data_list)):
    class_data_list.append(Data_csv(data_list[i], pd.read_csv(data_list[i])))
    class_data_list[i].data.drop(columns = ['Unnamed: 0'], inplace = True)
    if class_data_list[i].name[class_data_list[i].name.index('\\')+2] == 'C':
        class_data_list[i].data['Current'] = int(class_data_list[i].name[class_data_list[i].name.index('\\')+1])
    else:
        class_data_list[i].data['Current'] = int(class_data_list[i].name[5:7]) 
    
full_data = pd.DataFrame()
X = pd.DataFrame()
y = pd.DataFrame()

i = 0
for i in range(len(class_data_list)):

    full_data = pd.concat([full_data, class_data_list[i].data], ignore_index=True)

X = full_data.drop(columns=['y'])
y = full_data['y']

train_X, test_X = train_test_split(X, test_size=0.2, random_state=42)
train_y, test_y = train_test_split(y, test_size=0.2, random_state=42)

model=tf.keras.models.Sequential([

    tf.keras.layers.Dense(512),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(1)
           ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.mae)

t0 = perf_counter()
history = model.fit(x = train_X, y = train_y, epochs=100)
print(perf_counter() - t0)

test_loss = model.evaluate(test_X,test_y)

print("summary")
print(model.summary())

plt.plot(history.epoch,history.history['loss'])
plt.title('Loss versus Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.xticks(np.arange(0,101,5))
plt.scatter(100,test_loss,color='r')
plt.legend(('Train', 'Test'))
plt.show()