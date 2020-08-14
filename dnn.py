from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import re
from keras.utils import to_categorical
from keras.models import load_model

data_path = "/var/dataset/2020-6-29.dat"
test_path = "/var/dataset/2020-6-27.dat"

data_dim = 22
timesteps = 60
num_classes = 2

start_from = 18
end_from = 23


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
x_t0, y_t0 = getData('/var/dataset/2020-6-27.dat')
x_t1, y_t1 = getData('/var/dataset/2020-6-28.dat')
x_t2, y_t2 = getData('/var/dataset/2020-6-29.dat')
x_t3, y_t3 = getData('/var/dataset/2020-6-30.dat')
x_test, y_test = getData('/var/dataset/2020-7-1.dat')

x_t = [x_t0, x_t1, x_t2, x_t3]
y_t = [y_t0, y_t1, y_t2, y_t3]

model = load_model('./dnn.h5')



for i in range(4):

    model.fit(x_t[i], y_t[i],
        epochs=200,
        batch_size=128, validation_data=(x_test, y_test))

    model.save('./dnn.h5')


#score = model.evaluate(x_test, y_test, batch_size=128)
