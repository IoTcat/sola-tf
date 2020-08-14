from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import re
from keras.utils import to_categorical


data_path = "/var/dataset/2020-6-29.dat"
test_path = "/var/dataset/2020-6-27.dat"

data_dim = 20
timesteps = 8
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
        x.append(item[:20])
        y.append(item[-4:])


    x_data = []
    y_data = []


    for index in range(len(x) - timesteps - 2):
        tmp_sec_block = timesteps
        tmp_array = []
        while tmp_sec_block > 0:
            tmp_array.append(x[index + tmp_sec_block - 1])
            tmp_sec_block = tmp_sec_block - 1
        x_data.append(tmp_array)



    for index in range(len(y) - timesteps - 2):
        tmp = y[index + timesteps - 1]
        tmp = list(map(int, tmp))
        y_data.append(tmp[3])

    return np.array(x_data), np.array(y_data)














# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(1, activation='softmax'))

model.compile(loss='hinge',#'categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

# Generate dummy training data
x_train, y_train = getData(data_path)

print(x_train[0])
print(y_train[0])

# Generate dummy validation data
#x_val = np.random.random((100, timesteps, data_dim))
#y_val = np.random.random((100, num_classes))
x_val, y_val = getData(test_path)

model.fit(x_train, y_train,
        batch_size=64, epochs=5,
        validation_data=(x_val, y_val))



