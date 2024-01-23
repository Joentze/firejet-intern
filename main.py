import os
import time
import pickle
from typing import List
import cv2
import numpy as np
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img,   ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import legacy


def load_images(raw_path: str) -> tuple:
    """loads images and returns train set, label"""
    tags = os.listdir(raw_path)
    train_set = []
    label_set = []
    for tag in tags:
        files = os.listdir(f"{raw_path}/{tag}")
        for filename in files:
            print(f"{raw_path}/{tag}/{filename}")
            """load in image pixel array"""
            image = cv2.imread(f"{raw_path}/{tag}/{filename}")
            """reshape image"""
            image = cv2.resize(image, [256, 256])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            train_set.append(img_to_array(image))
            label_set.append(tag)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(label_set)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return np.array(train_set), onehot_encoded


def train_inspector_v3_model(train_path):
    """trains inspector model"""
    try:
        train_set, label_set = load_images(train_path)
        x_train, x_validation, y_train, y_validation = train_test_split(train_set, label_set, test_size=0.2,
                                                                        random_state=100)
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=30,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           horizontal_flip='true')

        train_generator = train_datagen.flow(
            x_train, y_train, shuffle=False, batch_size=40, seed=42)

        # Validation Generator
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        val_generator = val_datagen.flow(
            x_validation, y_validation, shuffle=False, batch_size=40, seed=42)

    # Model Intialize
        base_model = InceptionV3(
            weights='imagenet', include_top=False, input_shape=(256, 256, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        predictions = Dense(11, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(legacy.SGD(),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        start = time.time()
        model.fit_generator(train_generator,
                            steps_per_epoch=len(x_train) // 40,
                            validation_data=val_generator,
                            validation_steps=len(x_validation) // 40,
                            epochs=5,
                            verbose=2)
        end = time.time()
        print("\nTotal Time Taken:", round((end - start) / 60, 2), "Minutes")

        try:
            file = open("multi_class_model.pkl", "wb")
            pickle.dump(model, file)
            print("Model Saved..!!")

        except Exception as e:
            print(str(e))

    except Exception as e:
        print(str(e))


def predict_from_image(img_path: str, pickle_filename: str):
    model_classes = pickle.load(open(pickle_filename, "rb"))
    print(model_classes.summary())
    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels),
    # add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.
    pred = model_classes.predict(img_tensor)
    # sorted_category_list = sorted(top_categories)
    # one hot encoding is already alphabetically sorted from folder 
    predicted_class = ["a", "button", "footer", "form", "h1", "h2",
                       "h3", "h4", "header", "input", "textarea"][np.argmax(pred)]
    return predicted_class, max(pred[0])


if __name__ == "__main__":
    # train_inspector_v3_model("./raw/train")
    val = predict_from_image("./raw/test/12715.png", "multi_class_model.pkl")
    print(val)
