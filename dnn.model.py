from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import re
from keras.utils import to_categorical


data_path = "/var/dataset/2020-6-29.dat"
test_path = "/var/dataset/2020-6-27.dat"

data_dim = 22
timesteps = 60
num_classes = 2

start_from = 19
end_from = 19


def getData(path):

    data = []
    for line in open(path):
        tmp = re.findall(r"\d+\.?\d*",line)
        if int(tmp[20]) >= start_from:
            if int(tmp[20]) <= end_from:
                data.append(tmp)


    x = []
    y = []

    for item in data:
        x.append(item[:data_dim])
        y.append(item[-4:])


    x_data = []
    y_data = []


    for index in range(len(x) - timesteps - 2):
        tmp_sec_block = timesteps
        tmp_array = []
        while tmp_sec_block > 0:
            tmp_array += x[index + tmp_sec_block - 1]
            tmp_sec_block = tmp_sec_block - 1
        x_data.append(tmp_array)





    for index in range(len(y) - timesteps - 2):
        tmp = y[index + timesteps - 1]
        tmp = list(map(int, tmp))
        y_data.append(tmp[3])

    return np.array(x_data), np.array(y_data)



# Generate dummy training data
x_train, y_train = getData(data_path)
x_test, y_test = getData(test_path)

model = Sequential()
model.add(Dense(64, input_dim=data_dim*timesteps, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
model.save('./dnn.h5')


model.fit(x_train, y_train,
    epochs=50,
    batch_size=128, validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, batch_size=128)
