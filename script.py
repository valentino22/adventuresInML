# import os
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import regularizers
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # for a MacOs bug

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[0:30000]  # use a smaller subset for faster run
y_train = y_train[0:30000]  # use a smaller subset for faster run

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

image_height, image_width = 28, 28

X_train = X_train.reshape(30000, image_height * image_width)
X_test = X_test.reshape(10000, image_height * image_width)
print(X_train.shape)
print(X_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.0
X_test /= 255.0

print(y_train.shape)
print(y_test.shape)
print(y_train[0])

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(y_train.shape)
print(y_test.shape)
print(y_train[0])

model = Sequential()

model.add(Dense(512, activation='sigmoid', input_shape=(784,), activity_regularizer=regularizers.l2(0.0001),
                kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

plt.figure(1)
# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

score = model.evaluate(X_test, y_test)
print(score)
