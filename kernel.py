import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import cv2
from tqdm import tqdm

from sklearn.metrics import fbeta_score

input_size = 224
input_channels = 3

epochs = 100
batch_size = 128

model = Sequential()

model.add(Conv2D(16, kernel_size=(2, 2), padding='same', activation='relu',
                 input_shape=(input_size, input_size, input_channels)))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu',))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(17, activation='sigmoid'))

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=15,
                           verbose=0),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=5,
                               verbose=1)]

opt = Adam(lr=0.001)

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer=opt,
              metrics=['accuracy'])

df_train_data = pd.read_csv('input/train_v2.csv')

# Use 10% of data for hyper-parameter optimization
df_train_data = df_train_data[:int(len(df_train_data) * 0.1)]

print(len(df_train_data))

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# Calculate mean and stddev of dataset
mean_sum = np.zeros(3, dtype=np.float32)
stddev_sum = np.zeros(3, dtype=np.float32)

for f, tags in tqdm(df_train_data.values, miniters=1000):
    img = cv2.imread('input/train-jpg/{}.jpg'.format(f))
    mean, stddev = cv2.meanStdDev(src=img)
    mean_sum[0] += mean[0]
    mean_sum[1] += mean[1]
    mean_sum[2] += mean[2]
    stddev_sum[0] += stddev[0]
    stddev_sum[1] += stddev[1]
    stddev_sum[2] += stddev[2]

mean = mean_sum / len(df_train_data)
stddev = stddev_sum / len(df_train_data)


def normalize(src):
    src[:, :, 0] -= mean[0]
    src[:, :, 1] -= mean[1]
    src[:, :, 2] -= mean[2]
    src[:, :, 0] /= stddev[0]
    src[:, :, 1] /= stddev[1]
    src[:, :, 2] /= stddev[2]
    return src


def crop(src, x):
    return src[x:x+input_size, x:x+input_size, :]

validation_data_size = int(len(df_train_data) * 0.2)

print(validation_data_size)

df_valid = df_train_data[(len(df_train_data) - validation_data_size):]

x_valid = []
y_valid = []


for f, tags in tqdm(df_valid.values, miniters=100):
    img = cv2.imread('input/train-jpg/{}.jpg'.format(f)).astype(np.float32)
    img = normalize(img)
    x = ((256 - input_size) / 2) - 1
    img = crop(img, x)
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_valid.append(img)
    y_valid.append(targets)

y_valid = np.array(y_valid, np.uint8)
x_valid = np.array(x_valid, np.float32)

df_train = df_train_data[:(len(df_train_data) - validation_data_size)]


def train_generator():

    def transformations(src):
        # Random crop
        x = np.random.randint(0, 256 - input_size)
        src = crop(src, x)
        choice = np.random.randint(4)
        # Random horizontal flip
        if choice == 1:
            src = cv2.flip(src=src, flipCode=1)
        # Random vertical flip
        if choice == 2:
            src = cv2.flip(src=src, flipCode=0)
        if choice == 3:
            src = cv2.rotate(src=src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            choice = np.random.randint(3)
            # Random horizontal flip
            if choice == 1:
                src = cv2.flip(src=src, flipCode=1)
            # Random vertical flip
            if choice == 2:
                src = cv2.flip(src=src, flipCode=0)
        return src

    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    while True:
        for start in range(0, len(df_train), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(df_train))
            df_train_batch = df_train[start:end]
            for f, tags in df_train_batch.values:
                img = cv2.imread('input/train-jpg/{}.jpg'.format(f)).astype(np.float32)
                img = normalize(img)
                targets = np.zeros(17)
                for t in tags.split(' '):
                    targets[label_map[t]] = 1
                img = transformations(img)
                x_batch.append(img)
                y_batch.append(targets)
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.uint8)
            x_batch, y_batch = shuffle_in_unison(x_batch, y_batch)
            yield x_batch, y_batch


model.fit_generator(generator=train_generator(),
                    steps_per_epoch=(len(df_train) // batch_size) + 1,
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=(x_valid, y_valid))

df_test_data = pd.read_csv('input/sample_submission_v2.csv')


def test_generator():
    while True:
        for start in range(0, len(df_test_data), batch_size):
            x_batch = []
            end = min(start + batch_size, len(df_test_data))
            df_test_batch = df_test_data[start:end]
            for f, tags in df_test_batch.values:
                img = cv2.imread('input/test-jpg/{}.jpg'.format(f)).astype(np.float32)
                img = normalize(img)
                x = ((256 - input_size) / 2) - 1
                img = crop(img, x)
                x_batch.append(img)
            x_batch = np.array(x_batch, np.float32)
            yield x_batch


p_valid = model.predict(x_valid, batch_size=batch_size)
print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

y_test = []

p_test = model.predict_generator(generator=test_generator(),
                                 steps=(len(df_test_data) // batch_size) + 1)
y_test.append(p_test)

result = np.array(y_test[0])
result = pd.DataFrame(result, columns=labels)

preds = []

for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test_data['tags'] = preds
df_test_data.to_csv('submission.csv', index=False)

# 0.923 LB
