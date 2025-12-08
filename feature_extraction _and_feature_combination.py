#import os
#import cv2
#import numpy as np
#import tensorflow as tf
#from glob import glob
#from keras.utils import Sequence
#from keras.models import load_model
#from keras.models import Model
#from project import ViT, ClassToken  # Ensure these are correctly imported according to your project structure
#
## Define hyperparameters
#hp = {
#    "image_size": 300,
#    "num_channels": 3,
#    "patch_size": 10,
#    "num_classes": 4,
#    "class_names": ["Mayo0", "Mayo1", "Mayo2", "Mayo3"],
#    "num_layers": 4,
#    "hidden_dim": 768,
#    "mlp_dim": 3072,
#    "num_heads": 4,
#    "dropout_rate": 0.1,
#    "batch_size": 1,
#    "lr": 1e-4,
#    "num_epochs": 3,
#}
#
#hp["num_patches"] = (hp["image_size"] ** 2) // (hp["patch_size"] ** 2)
#hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"] * hp["patch_size"] * hp["num_channels"])
#
## Define paths
#dataset_path = '/home/nehal/workspace/cancer_classification/New folder (3)'
#model_path = 'New folder/modelN_fold1.h5'
#
#def create_dir(path):
#    if not os.path.exists(path):
#        os.makedirs(path)
#
#create_dir("files")
#
#def process_image(file_path):
#    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#    image = cv2.resize(image, (hp['image_size'], hp['image_size']))
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    image = image / 255.0
#    patches = np.reshape(image, hp["flat_patches_shape"]).astype(np.float32)  # Changed to float32 for better precision
#    return patches
#
#class DataGenerator(Sequence):
#    def __init__(self, image_paths, labels, batch_size):
#        self.image_paths = image_paths
#        self.labels = labels
#        self.batch_size = batch_size
#
#    def __len__(self):
#        return int(np.ceil(len(self.image_paths) / self.batch_size))
#
#    def __getitem__(self, index):
#        batch_indices = slice(index * self.batch_size, (index + 1) * self.batch_size)
#        batch_paths = self.image_paths[batch_indices]
#        batch_labels = self.labels[batch_indices]
#        X = np.array([process_image(path) for path in batch_paths])
#        y = tf.keras.utils.to_categorical(batch_labels, num_classes=hp["num_classes"])
#        return X, y
#
### Load and instantiate the model
#custom_objects = {"ClassToken": ClassToken}
#model = load_model(model_path, custom_objects=custom_objects)
#print(model.summary())
#feature_extraction_model = Model(inputs=model.input, outputs=model.get_layer('tf.__operators__.getitem').output)
#
## Process images and save features in batches
#image_paths = glob(os.path.join(dataset_path, "*", "*.jpg"))
#labels = [hp["class_names"].index(os.path.basename(os.path.dirname(path))) for path in image_paths]
#
#total_images = len(image_paths)
#batch_size = 12134  # Number of images to process at a time
#
#for start_idx in range(0, total_images, batch_size):
#    end_idx = start_idx + batch_size
#    batch_paths = image_paths[start_idx:end_idx]
#    batch_labels = labels[start_idx:end_idx]
#    generator = DataGenerator(batch_paths, batch_labels, hp["batch_size"])
#    features = feature_extraction_model.predict(generator, verbose=1)
#    np.save(f'extracted_features_label{start_idx//batch_size}.npy', features)
#
#print("Feature extraction complete.")
#
#
#import os
#import numpy as np
#from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import to_categorical
#from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
#import tensorflow as tf
## Set the path to the test directory
#base_dir = '/home/nehal/workspace/cancer_classification/train/New folder'
#test_dir = os.path.join(base_dir)
#
## Load the complete model
##model = load_model('best_model.h5')
##print(model.summary())
## Generate predictions
##predictions = model.predict(test_generator)
##predicted_classes = np.argmax(predictions, axis=1)
##true_classes = test_generator.classes
##class_labels = list(test_generator.class_indices.keys())
#
## Output the confusion matrix and classification report
##print("Confusion Matrix:")
##print(confusion_matrix(true_classes, predicted_classes))
##print("Classification Report:")
##print(classification_report(true_classes, predicted_classes, target_names=class_labels))
#
### Calculate and display the AUC-ROC Score
##predicted_probs = model.predict(test_generator)
##roc_score = roc_auc_score(to_categorical(true_classes), predicted_probs, multi_class='ovr')
##print(f"AUC-ROC Score: {roc_score}")
## Prepare the test data generator
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#    except RuntimeError as e:
#        print(e)
#test_datagen = ImageDataGenerator(rescale=1./255)
#test_generator = test_datagen.flow_from_directory(
#    test_dir, 
#    target_size=(150, 150), 
#    batch_size=512,  # Your specified batch size
#    class_mode='categorical', 
#    shuffle=False
#)
#
## Load your model
#model = load_model('best_model.h5')
##print(model.summary())
#
## Predict and save features in separate files for each batch
#number_of_batches = len(test_generator)
#for batch_num in range(number_of_batches):
#    # Retrieve the data of the current batch
#    x_batch, y_batch = next(test_generator)
#    
#    # Predict the features for the current batch
#    features_batch = model.predict(x_batch, verbose=1)
#    
#    # Save the features of the current batch to a .npy file
#    batch_file_name = f'extracted_features_renet_50_batch_{batch_num}.npy'
#    np.save(batch_file_name, features_batch)
#    print(f"Batch {batch_num + 1}/{number_of_batches} features saved to {batch_file_name}")
##import os
#import numpy as np
#from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import to_categorical
#from keras.models import Model
#
## ... (Your code for loading the test data generator) 
#
## Load your model
#model = load_model('best_model.h5')
#print(model.summary())
## Create an intermediate model for feature extraction
#feature_extraction_model = Model(inputs=model.input,
#                                 outputs=model.get_layer("dense").output) 
#
## Extract the features 
#features = feature_extraction_model.predict(test_generator, verbose=1)
## Save the features
#np.save('extracted_features_renet_50.npy', features) 
#import os
#import cv2
#import numpy as np
#import tensorflow as tf
#from glob import glob
#from keras.models import load_model
#from sklearn.metrics import classification_report, roc_auc_score, roc_curve
#import matplotlib.pyplot as plt
#from project import ViT, ClassToken  # Ensure these are correctly imported according to your project structure
#from keras.models import Model
# Define hyperparameters and setup
#hp = {
##    "image_size": 150,        # Size of the images (150x150 pixels)
##    "num_channels": 3,        # Number of color channels (RGB)
##    "patch_size": 10,         # Size of the patches each image is divided into (10x10 pixels)
##    "num_classes": 4,         # Number of classes in the dataset
##    "class_names": ["BKL", "MEL2", "NV", "BCC"],  # Names of the classes
##    "num_layers": 4,          # Number of layers in the ViT model
##    "hidden_dim": 512,        # Dimensionality of the encoder layers
##    "mlp_dim": 3072,          # Dimensionality of the dense layers of the model
##    "num_heads": 4,           # Number of attention heads
##    "dropout_rate": 0.1,      # Dropout rate
##    "batch_size": 2,         # Batch size for training
##    "lr": 1e-4,               # Learning rate
##    "num_epochs": 3           # Number of epochs for training
##}
#hp = {
#    "image_size": 300,
#    "num_channels": 3,
#    "patch_size": 10,
#    "num_classes": 4,
#    "class_names": ["Mayo0", "Mayo1", "Mayo2", "Mayo3"],
#    #"class_names": ["BKL","MEL2", "NV","BCC"],
#    "num_layers": 4,
#    "hidden_dim": 768,
#    "mlp_dim": 3072,
#    "num_heads": 4,
#    "dropout_rate": 0.1,
#    "batch_size": 4,
#    "lr": 1e-4,
#    "num_epochs": 3,
#}
#
## Calculating additional hyperparameters
#hp["num_patches"] = (hp["image_size"] ** 2) // (hp["patch_size"] ** 2)  # Compute the number of patches per image
#hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"] * hp["patch_size"] * hp["num_channels"])  # Compute the flattened shape of patches
#
## Define paths and create directories if necessary
##dataset_path = '/home/nehal/workspace/cancer_classification/train/New folder'
#dataset_path = '/home/nehal/workspace/cancer_classification/New folder (3)'
#model_path = 'New folder/modelN_fold4.h5'
#  # Use float32 instead of the default float64
#
#def create_dir(path):
#    """ Create directory if it does not exist. """
#    if not os.path.exists(path):
#        os.makedirs(path)
#
#create_dir("files")
#
#def process_image(file_path):
#    """ Process images to the format required by the ViT model. """
#    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#    image = cv2.resize(image, (hp['image_size'], hp['image_size']))
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    image = image / 255.0
#    patches = np.reshape(image, hp["flat_patches_shape"]).astype(np.float16)
#    return patches
#
#def load_and_process_data(path):
#    """ Load and process data for model evaluation. """
#    images = glob(os.path.join(path, "*", "*.jpg"))
#    data = np.array([process_image(img) for img in images])
#    labels = [hp["class_names"].index(img.split('/')[-2]) for img in images]
#    return data, labels
#
## Load test data
#test_data, test_labels = load_and_process_data(dataset_path)
#
## Convert labels to one-hot for compatibility with categorical crossentropy
#test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=hp["num_classes"])
#
## Load model with custom components
#custom_objects = {"ClassToken": ClassToken}  # Add custom objects if your ViT implementation uses any
#model = load_model(model_path, custom_objects=custom_objects)
#print(model.summary())
#target_layer_name = 'tf.__operators__.getitem_3'
#feature_extraction_model = Model(inputs=model.input,
#                                 outputs=model.get_layer(target_layer_name).output)
##
## Extract and save features (rest of the process remains the same)
#features = feature_extraction_model.predict(test_data, batch_size=hp["batch_size"])
#np.save('extracted_features_vitty.npy', features) 
##print(model.summary())
## Model evaluation
##predictions = model.predict(test_data, batch_size=hp["batch_size"])
##predicted_classes = np.argmax(predictions, axis=1)
##
### Classification report
##print("Classification Report:")
##print(classification_report(test_labels, predicted_classes, target_names=hp["class_names"]))
##
### Compute AUC-ROC per category
##print("AUC-ROC Scores:")
##for i, class_name in enumerate(hp["class_names"]):
##    roc_auc = roc_auc_score(test_labels_one_hot[:, i], predictions[:, i])
##    print(f"{class_name}: {roc_auc:.4f}")
##
#    # Optional: Plot ROC Curve per class
#    fpr, tpr, _ = roc_curve(test_labels_one_hot[:, i], predictions[:, i])
#    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
#
#plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.50)")
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve per Class')
#plt.legend(loc="lower right")
#plt.show()

import numpy as np
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Set base directory where the class directories are located
base_dir = '/home/nehal/workspace/cancer_classification/train/New folder'

# Load extracted features
features_resnet = np.load('/home/nehal/workspace/cancer_classification/combined_features22.npy')
print("/SIDE+REAR/Shape of features_resnet:", features_resnet.shape)

# Assume combined_features is just features_resnet for now
combined_features = features_resnet
print("Shape of combined features:", combined_features.shape)

# Class names and corresponding labels
class_names = ["BKL", "MEL2", "NV", "BCC"]
class_labels = {class_name: i for i, class_name in enumerate(class_names)}

# Count the number of images per class directory accurately
num_images_per_class = []
for class_name in class_names:
    class_dir = os.path.join(base_dir, class_name)
    # Count only files, assume images
    image_files = [name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))]
    num_images_per_class.append(len(image_files))

print("Images per class:", num_images_per_class)

# Generate the label array with the correct number of labels per class
labels = np.hstack([[class_labels[class_name]] * num for class_name, num in zip(class_names, num_images_per_class)])
print("Total labels:", len(labels))

# Convert labels to one-hot encoding
labels_one_hot = to_categorical(labels, num_classes=len(class_names))
print("Labels one-hot shape:", labels_one_hot.shape)

# Ensure the number of samples in features and labels are the same before splitting
if combined_features.shape[0] == labels_one_hot.shape[0]:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(combined_features, labels_one_hot, test_size=0.2, random_state=42)
    print("Data split successful. Training data shape:", X_train.shape)
else:
    print("Mismatch in number of samples between features and labels:", combined_features.shape[0], labels_one_hot.shape[0])

# Define the neural network model
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(len(class_names), activation='softmax')
])

# You would typically compile and train the model here
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)


## Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
## Fit the model
#history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.1, verbose=1)
#
## Evaluate the model
#test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#print(f'Test Accuracy: {test_acc * 100:.2f}%')
#import numpy as np
#import os
#from keras.utils import to_categorical
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.callbacks import ModelCheckpoint
#import tensorflow as tf
#import numpy as np
#from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVC
#from sklearn.metrics import classification_report, accuracy_score
#from sklearn.utils import shuffle
#
## Load combined features and labels
#features = np.load('/home/nehal/workspace/cancer_classification/combined_features_abdulla.npy')
#labels = np.load('/home/nehal/workspace/cancer_classification/combined_features_label.npy')
#features, labels = shuffle(features, labels, random_state=42)
#
## Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#
## Normalize the feature vectors
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
#
## Train the SVM classifier
#svm_classifier = SVC(kernel='linear')
#svm_classifier.fit(X_train_scaled, y_train)
#
## Predict on the test set
#y_pred = svm_classifier.predict(X_test_scaled)
#
## Evaluate the classifier
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:")
#print(classification_report(y_test, y_pred))
#


#
#base_dir = '/home/nehal/workspace/cancer_classification/SIDE+REAR'
#
## Load and combine features from each directory
#features_folder_1 = load_and_combine_features(os.path.join(base_dir, 'New folder'))
#features_folder_2 = load_and_combine_features(os.path.join(base_dir, 'New folder2'))
#features_folder_3 = load_and_combine_features(os.path.join(base_dir, 'New folder3'))
#features_folder_4 = load_and_combine_features(os.path.join(base_dir, 'New folder4'))
#
## Ensure all feature sets are of the same length
#min_size = min(features_folder_1.shape[0], features_folder_2.shape[0], features_folder_3.shape[0], features_folder_4.shape[0])
#features_folder_1 = features_folder_1[:min_size]
#features_folder_2 = features_folder_2[:min_size]
#features_folder_3 = features_folder_3[:min_size]
#features_folder_4 = features_folder_4[:min_size]
#
## Concatenate the features from all folders along axis 1
#final_combined_features = np.concatenate([features_folder_1, features_folder_2, features_folder_3, features_folder_4], axis=1)
#output_file = '/home/nehal/workspace/cancer_classification/combined_features_label.npy'
##
#np.save(output_file, final_combined_features)
### Print the shape of the final combined features to verify
#print("Shape of final combined features:", final_combined_features.shape)
#
# Configuring GPU memory usage
#    print(f"ROC-AUC Score for {class_name}: {roc_auc_scores[i]:.2f}")

# Split the dataset
#X_train, X_test, y_train, y_test = train_test_split(combined_features, labels_one_hot, test_size=0.2, random_state=42)

# Define and compile the neural network model
#model = Sequential([
#    Dense(1024, activation='relu', input_dim=2304),#1536),
#    Dense(len(class_names), activation='softmax')
#])
#
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
## Set up a model checkpoint
#checkpoint = ModelCheckpoint('the_best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
#
## Train the model
#history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_split=0.1, callbacks=[checkpoint], verbose=1)
#
## Evaluate the model on the test set
#test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#print(f"Test Accuracy: {test_acc * 100:.2f}%")
#
## Calculate and print ROC-AUC for each class
#y_pred = model.predict(X_test)
#roc_auc_scores = roc_auc_score(y_test, y_pred, multi_class='ovr', average=None)  # 'ovr' means One-vs-Rest
#for i, class_name in enumerate(class_names):
#    print(f"ROC-AUC Score for {class_name}: {roc_auc_scores[i]:.2f}")
##
### Save the final model
#model.save('final_model.h5')
#import numpy as np
#import os
#from keras.utils import to_categorical
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score
#from keras.models import Model
#from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout
#from keras.callbacks import ModelCheckpoint
#import tensorflow as tf
#import cv2
#def load_images_from_folder(folder, image_size=(150, 150)):
#    images = []
#    labels = []
#    label_map = {class_folder: i for i, class_folder in enumerate(sorted(os.listdir(folder)))}
#    for class_folder in sorted(os.listdir(folder)):
#        class_path = os.path.join(folder, class_folder)
#        for image_file in sorted(os.listdir(class_path)):
#            image_path = os.path.join(class_path, image_file)
#            image = cv2.imread(image_path)
#            image = cv2.resize(image, image_size)
#            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#            images.append(image)
#            labels.append(label_map[class_folder])
#    images = np.array(images, dtype=np.float32) / 255.0  # Normalize the images
#    labels = np.array(labels)
#    return images, labels
## Configuring GPU memory usage
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        tf.config.experimental.set_virtual_device_configuration(
#            gpus[0],
#            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 5.5)])  # Limit to 5.5GB
#    except RuntimeError as e:
#        print("Runtime error: ", e)
#
## Set the base directory where the data is located
#base_dir = '/home/nehal/workspace/cancer_classification/train/New folder'
#image_dir = base_dir  # Update if your images are in a subdirectory
#
## Load and preprocess the images
#X_images, y_images = load_images_from_folder(image_dir)
#y_images_one_hot = to_categorical(y_images)
#
## Load pre-extracted features (replace with actual paths)
#features_resnet = np.load('path_to_features_resnet.npy')
#features_vgg16 = np.load('path_to_features_vgg16.npy')
#features_vit = np.load('path_to_features_vit.npy')
#
## Align features with images (assuming they are in the same order)
#min_size = min(len(X_images), features_resnet.shape[0], features_vgg16.shape[0], features_vit.shape[0])
#X_images = X_images[:min_size]
#y_images_one_hot = y_images_one_hot[:min_size]
#X_features = np.concatenate([features_resnet[:min_size], features_vgg16[:min_size], features_vit[:min_size]], axis=1)
#
## Split the dataset into training and testing sets
#X_train_images, X_test_images, X_train_features, X_test_features, y_train, y_test = train_test_split(
#    X_images, X_features, y_images_one_hot, test_size=0.2, random_state=42
#)
#
## Neural network with two branches
## Image processing branch
#image_input = Input(shape=X_train_images.shape[1:], name='image_input')
#x1 = Conv2D(32, (3, 3), activation='relu')(image_input)
#x1 = MaxPooling2D((2, 2))(x1)
#x1 = Conv2D(64, (3, 3), activation='relu')(x1)
#x1 = MaxPooling2D((2, 2))(x1)
#x1 = Flatten()(x1)
#
## Feature processing branch
#feature_input = Input(shape=(X_train_features.shape[1],), name='feature_input')
#x2 = Dense(1024, activation='relu')(feature_input)
#x2 = Dropout(0.5)(x2)
#x2 = Dense(512, activation='relu')(x2)
#x2 = Dropout(0.5)(x2)
#
## Combine branches
#combined = concatenate([x1, x2])
#combined = Dense(256, activation='relu')(combined)
#output = Dense(y_train.shape[1], activation='softmax')(combined)
#model = Model(inputs=[image_input, feature_input], outputs=output)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
#
## Set up model checkpointing
#checkpoint = ModelCheckpoint('best_combined_model.h5', monitor='val_auc', mode='max', save_best_only=True, verbose=1)
#
## Train the model
#history = model.fit(
#    [X_train_images, X_train_features], y_train,
#    validation_data=([X_test_images, X_test_features], y_test),
#    epochs=15, batch_size=16,
#    callbacks=[checkpoint],
#    verbose=1
#)
#
## Evaluate the model
#test_loss, test_acc, test_auc = model.evaluate([X_test_images, X_test_features], y_test, verbose=2)
#print(f"Test Accuracy: {test_acc * 100:.2f}%, Test AUC: {test_auc:.2f}")
#
## Save the final model
#model.save('final_combined_model.h5')
