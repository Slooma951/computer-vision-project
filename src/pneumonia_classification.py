from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.optimizers import RMSprop, Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

batch_size = 12
num_classes = 3
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True  # Make fit false if you do not want to train the network again
train_dir = '../data/train'
test_dir = '../data/test'

with tf.device('/gpu:0'):

    # Create training, validation, and test datasets
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    class_names = train_ds.class_names
    print('Class Names: ', class_names)
    num_classes = len(class_names)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Create the model
    model = tf.keras.models.Sequential([
        Rescaling(1.0/255),  # Rescale pixel values
        data_augmentation,  # Apply augmentation
        Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # EarlyStopping callback to avoid overfitting
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Save the best model
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras", save_freq='epoch', save_best_only=True)

    # Fit the model
    if fit:
        history = model.fit(
            train_ds,
            batch_size=batch_size,
            validation_data=val_ds,
            callbacks=[save_callback, earlystop_callback],
            epochs=epochs)
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    # Evaluate the model
    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    # Plot accuracy vs. validation accuracy
    if fit:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

    # Show predictions on some test images
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(), 0))
            plt.title('Actual:' + class_names[labels[i].numpy()] + '\nPredicted:{} {:.2f}%'.format(
                class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.show()

    # Classification report (precision, recall, F1-score)
    predictions = model.predict(test_ds)
    y_pred = np.argmax(predictions, axis=1)
    y_true = []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))