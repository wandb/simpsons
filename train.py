from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import subprocess
import os

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config
config.img_size = 100
config.batch_size = 64
config.epochs = 25

# download the data if it doesn't exist
if not os.path.exists("simpsons"):
    print("Downloading Simpsons dataset...")
    subprocess.check_output(
        "curl https://storage.googleapis.com/wandb-production.appspot.com/mlclass/simpsons.tar.gz | tar xvz", shell=True)

# this is the augmentation configuration we will use for training
# see: https://keras.io/preprocessing/image/#imagedatagenerator-class
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'simpsons/train',
    target_size=(config.img_size, config.img_size),
    batch_size=config.batch_size)
test_generator = test_datagen.flow_from_directory(
    'simpsons/test',
    target_size=(config.img_size, config.img_size),
    batch_size=config.batch_size)

labels = list(test_generator.class_indices.keys())

model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(13, activation="softmax"))
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator) // config.batch_size,
    epochs=config.epochs,
    workers=4,
    validation_data=test_generator,
    callbacks=[WandbCallback(
        data_type="image", labels=labels, generator=test_generator, save_model=False)],
    validation_steps=len(test_generator) // config.batch_size)
