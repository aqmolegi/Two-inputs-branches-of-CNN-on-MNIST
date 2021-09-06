from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os


(x_train, y_train),(x_test,y_test) = mnist.load_data()

lbl_unique = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_shape = (image_size, image_size, 1)

x_train = np.reshape(x_train, [-1,image_size,image_size,1]) / 255
x_test = np.reshape(x_test, [-1,image_size,image_size,1]) / 255 

filters = 200
kernel_size = 3
batch_size = 200
dropout = 0.2

model_path = './save_model'
if not os.path.exists(model_path):
    os.mkdir(model_path)

model_checkpoint = ModelCheckpoint(filepath = model_path +'/CNN_2Branches.h5',
                                   monitor = 'accuracy',
                                   mode='max',
                                   save_best_only = True,
                                   save_weights_only=False,
                                   verbose = 1)
early_stopping = EarlyStopping(monitor = 'accuracy', patience = 10)
call_backs = [model_checkpoint,early_stopping]


Left_D_input = Input(shape = input_shape)
for i in range(3):
    L = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', activation = 'relu', dilation_rate = 2) (Left_D_input)
    L = Dropout(dropout)(L)
    L = MaxPooling2D(pool_size = 2)(L)

Right_D_input = Input(shape = input_shape)
for i in range(3):
    R = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', activation = 'relu') (Right_D_input)
    R = Dropout(dropout)(R)
    R = MaxPooling2D(pool_size = 2)(R)

LR = concatenate([L,R]) # merge left and right branches outputs
LR = Flatten()(LR)
LR = Dropout(dropout)(LR)
D_output = Dense(lbl_unique, activation = 'softmax')(LR)

model = Model([Left_D_input, Right_D_input], D_output)
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit([x_train,x_train], y_train, epochs = 10, batch_size = batch_size, callbacks = call_backs)
_, acc = model.evaluate([x_test,x_test], y_test, batch_size = batch_size, verbose = 0)

print("\nTesting accuracy: %.1f%%" % (100.0 * acc))