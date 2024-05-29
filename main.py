import os
import time
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
def plot_loss_accuracy(history, save_model):
    """
    Plots training and validation accuracy and loss from a Keras History object.

    Parameters:
    - history: Keras History object.
    - save_model: Boolean, if True saves the plots as PNG files.
    """
    plt.figure(figsize=(10, 6))

    plt.title("Model Accuracy")
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    if save_model:
        plt.savefig("acc-epoch.png", dpi=300)
    plt.show()

    plt.title("Model Loss")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    if save_model:
        plt.savefig("loss-epoch.png", dpi=300)
    plt.show()

def plot_confusion_matrix(model, generator):
    """
    Plots the confusion matrix for a classification model.

    Parameters:
    - model: Trained classification model.
    - generator: Data generator used for validation/testing.

    This function computes predictions on the validation/test set and then plots
    the confusion matrix using seaborn's heatmap.
    """
    y_val_pred = model.predict(generator, steps=generator.samples // generator.batch_size + 1)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    y_val_true = generator.classes
    cm = confusion_matrix(y_val_true, y_val_pred_classes)
    class_names = list(generator.class_indices.keys())
    plt.figure(figsize=(14, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def explain_evolution(first_class, second_class):
    """
    Provides explanations for land cover class evoluations.

    Parameters:
    - first_class: Integer representing the initial land cover class.
    - second_class: Integer representing the resulting land cover class.

    Returns:
    Explanation string for the transition between the specified classes.
    """
    explanations = {
        (1, 4): ("Forest areas were converted into industrial zones.\n"
                 "Ecosystem balance is disrupted, biodiversity is reduced, and climate change is accelerated.\n"
                 "Increased carbon emissions contribute to global warming.\n"
                 "Forest conservation and sustainable urbanization policies should be developed."),
        (1, 3): ("Forest areas were converted into highways.\n"
                 "Natural habitats are destroyed and biodiversity is decreased.\n"
                 "Deforestation contributes to global warming by reducing carbon sequestration.\n"
                 "Eco-friendly transportation solutions should be developed."),
        (1, 7): ("Forest areas were converted into residential zones.\n"
                 "Natural habitats are destroyed and biodiversity is decreased.\n"
                 "Deforestation contributes to global warming by reducing carbon sequestration.\n"
                 "Sustainable urban planning should be implemented."),
        (2, 3): ("Herbaceous vegetation areas were converted into highways.\n"
                 "Natural areas are fragmented and ecosystems are disrupted.\n"
                 "Increased vehicle emissions contribute to air pollution and global warming.\n"
                 "Eco-friendly transportation solutions should be developed."),
        (0, 6): ("Annual crop areas were transformed into permanent crop areas.\n"
                 "Agricultural production sustainability can be enhanced, but soil fertility should be maintained through appropriate farming techniques.\n"
                 "Sustainable farming practices can mitigate global warming impacts."),
        (5, 7): ("Pasture areas were converted into residential zones.\n"
                 "Livestock farming is impacted and open spaces are reduced.\n"
                 "Urbanization increases carbon footprint and contributes to global warming.\n"
                 "Sustainable development practices should be considered."),
        (4, 7): ("Industrial areas were converted into residential zones.\n"
                 "Urban redevelopment may be indicated.\n"
                 "Transitioning from industrial to residential zones can reduce local pollution but may increase overall carbon footprint due to increased housing.\n"
                 "Sustainable and community-focused redevelopment strategies should be implemented."),
        (2, 5): ("Herbaceous vegetation was transformed into pasture areas.\n"
                 "Livestock farming can be supported, but ecological balance should be maintained.\n"
                 "Sustainable grazing practices are essential to mitigate greenhouse gas emissions."),
        (1, 2): ("Forest areas were transformed into herbaceous vegetation.\n"
                 "Deforestation or natural succession might be the cause.\n"
                 "Loss of trees reduces carbon sequestration, contributing to global warming.\n"
                 "Efforts to restore forests should be prioritized."),
        (7, 8): ("Residential areas were transformed into sealake zones.\n"
                 "This unusual change could indicate natural disasters.\n"
                 "Climate change can increase the frequency of such events. Effective flood management and urban planning are required."),
        (0, 5): ("Annual crop areas were converted into pasture areas.\n"
                 "Agricultural land use was shifted, affecting crop production.\n"
                 "Sustainable land management practices should be employed to maintain soil health and mitigate climate change impacts."),
        (5, 0): ("Pasture areas were converted into annual crop areas.\n"
                 "Grazing land was shifted to crop production, impacting livestock farming.\n"
                 "Integrated land use strategies should be developed to balance food production and carbon emissions."),
        (6, 0): ("Permanent crop areas were converted into annual crop areas.\n"
                 "Long-term agricultural practices were replaced by short-term ones.\n"
                 "Changes in crop types can affect carbon sequestration. Agricultural sustainability practices should be reviewed."),
        (8, 7): ("River areas were converted into residential zones.\n"
                 "Water bodies were replaced by urban development, affecting natural water flow.\n"
                 "Urbanization can increase the risk of flooding and heat islands, exacerbating global warming. Urban planning must consider environmental impacts."),
        (7, 4): ("Residential areas were converted into industrial zones.\n"
                 "Urban living spaces were replaced by industrial development.\n"
                 "Industrial activities increase carbon emissions, contributing to global warming. Balance between residential and industrial areas should be maintained."),
        # Explanation for no change
        (0, 0): ("Annual crop areas remained unchanged. The area has been protected and no significant changes occurred."),
        (1, 1): ("Forest areas remained unchanged. The area has been protected and no significant changes occurred."),
        (2, 2): ("Herbaceous vegetation areas remained unchanged. The area has been protected and no significant changes occurred."),
        (3, 3): ("Highway areas remained unchanged. The area has been maintained and no significant changes occurred."),
        (4, 4): ("Industrial areas remained unchanged. The area has been maintained and no significant changes occurred."),
        (5, 5): ("Pasture areas remained unchanged. The area has been protected and no significant changes occurred."),
        (6, 6): ("Permanent crop areas remained unchanged. The area has been maintained and no significant changes occurred."),
        (7, 7): ("Residential areas remained unchanged. The area has been maintained and no significant changes occurred."),
        (8, 8): ("River areas remained unchanged. The area has been protected and no significant changes occurred."),
        (9, 9): ("Sea/lake areas remained unchanged. The area has been protected and no significant changes occurred.")

    }

    return explanations.get((first_class, second_class), "No significant change observed between the specified classes.")

def prepare_image_to_test(file_path, target_size=(64, 64)):
    """
    Prepares an image for testing by loading, resizing, and normalizing it.

    Parameters:
    - file_path: String representing the path to the image file.
    - target_size: Tuple specifying the target size for the image (default is (64, 64)).

    Returns:
    Numpy array representing the preprocessed image.
    """
    img = keras.preprocessing.image.load_img(file_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    plt.imshow(img)
    plt.title(f'Image: {file_path}')
    plt.axis('off')
    plt.show()

    return img_array

class ImageClassifierCallback(keras.callbacks.Callback):
    """
    Custom callback for monitoring training progress, stopping criteria and saving model.

    Parameters:
    - save_model: Boolean indicating whether to save the model checkpoints.
    """
    def __init__(self, save_model):
        super().__init__()
        self.save_model = save_model
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and logs.get("val_accuracy") > 0.955:
            print(f"\nReached 93% validation accuracy, training stopped..")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        training_duration = time.time() - self.start_time
        hours = training_duration // 3600
        minutes = (training_duration - (hours * 3600)) // 60
        seconds = training_duration - ((hours * 3600) + (minutes * 60))

        message = f"Training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds."
        print(message)

        if logs is not None and self.save_model:
            self.model.save(f"models/classifier_CNN{logs.get('accuracy') * 100:.2f}Acc.h5")
            print("The model is successfully saved.")

if __name__ == '__main__':
    # Assigning some sonstants
    BASE_DIR = "data/"
    CLASSES = os.listdir(BASE_DIR)
    NUM_CLASSES = len(CLASSES)
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 16
    EPOCHS = 100
    SAVE_MODEL = True
    USE_SAVED_MODEL = True
    SHOW_TRAINING_STATISTICS = not USE_SAVED_MODEL
    COLOR_MODE = "rgb"

    print(f"Contents of train dir: {CLASSES}")
    print("----------------------------------------")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )
    train_generator = datagen.flow_from_directory(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        shuffle=True,
        subset="training",
        color_mode=COLOR_MODE
    )
    validation_generator = datagen.flow_from_directory(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        shuffle=True,
        subset="validation",
        color_mode=COLOR_MODE
    )

    # Printing sample's shape
    sample_batch = next(train_generator)
    print(f"Shape of sample batch is : {sample_batch[0].shape}")
    print("----------------------------------------")

    class_counts = train_generator.classes
    unique_classes, class_counts = np.unique(class_counts, return_counts=True)

    # Checking whether the data is balanced or not
    for cls, count in zip(unique_classes, class_counts):
        print(f"There are {count} images for {CLASSES[cls]}.")
    print("----------------------------------------")

    # Plotting the first image of the training generator to getting insight
    first_img = sample_batch[0][0]
    img_label = CLASSES[np.argmax(sample_batch[1][0])]
    plt.imshow(first_img)
    plt.title("First image of training set (" + img_label + ")\n" + str(first_img.shape))
    plt.axis('off')
    plt.show()

    if USE_SAVED_MODEL:
        model = keras.models.load_model("models/classifier.h5")

        # Predicting labels of old and current images of the same area
        old_image_path = "test/amazon50YearsAgo.jpg"
        new_image_path = "test/amazonCurrent.jpg"
        old_image_array = prepare_image_to_test(old_image_path)
        new_image_array = prepare_image_to_test(new_image_path)

        # Probabilities
        old_image_probabilities = model.predict(old_image_array)[0]
        new_image_probabilities = model.predict(new_image_array)[0]

        # Getting top 3 class probability
        top_3_old_indices = np.argsort(old_image_probabilities)[-3:][::-1]
        top_3_new_indices = np.argsort(new_image_probabilities)[-3:][::-1]

        # Getting top 3 class names and formatted probabilities
        top_3_old_classes = ", ".join([f"{CLASSES[class_ix]}" for class_ix in top_3_old_indices])
        top_3_new_classes = ", ".join([f"{CLASSES[class_ix]}" for class_ix in top_3_new_indices])
        top_3_old_probs = ", ".join([f"{prob*100:.2f}%" for prob in old_image_probabilities[top_3_old_indices]])
        top_3_new_probs = ", ".join([f"{prob*100:.2f}%" for prob in new_image_probabilities[top_3_new_indices]])

        print("Indices of the 3 highest probabilities for the old image:", top_3_old_classes)
        print("Top 3 possibilities for old image:", top_3_old_probs)

        print("\nIndices of the 3 highest probabilities for the new image:", top_3_new_classes)
        print("Top 3 possibilities for new image:", top_3_new_probs)
        print("----------------------------------------")

        print(explain_evolution(top_3_old_indices[0], top_3_new_indices[0]))
    else:
        # Creating deep convolutional neural network architecture
        model = keras.models.Sequential([
            keras.layers.Input(shape=(64, 64, 1 if COLOR_MODE == "grayscale" else 3)),
            keras.layers.Conv2D(16, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(NUM_CLASSES, activation="softmax")
        ])

        # Compiling and setting the model
        model.compile(optimizer=keras.optimizers.Adamax(learning_rate=4e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())

        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[
                ImageClassifierCallback(SAVE_MODEL)
            ]
        )

        if SHOW_TRAINING_STATISTICS:
            # Showing statistics about training and the model
            plot_loss_accuracy(history, SAVE_MODEL)

    test_generator = datagen.flow_from_directory(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        shuffle=False,
        subset="validation",
        color_mode=COLOR_MODE
    )
    plot_confusion_matrix(model, test_generator)

# rapor + sunum hazırla