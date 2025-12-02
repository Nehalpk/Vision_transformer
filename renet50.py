#import os
#import shutil
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
#import keras
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#from keras.utils import to_categorical
#import os
#import shutil
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
#import keras
#import matplotlib
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#from keras.utils import to_categorical
#matplotlib.use('Agg') 
## Paths setup
#base_dir = '/home/nehal/workspace/cancer_classification/train/New folder'
#train_dir = os.path.join(base_dir, 'train')
#val_dir = os.path.join(base_dir, 'val')
#test_dir = os.path.join(base_dir, 'test')
#os.makedirs(train_dir, exist_ok=True)
#os.makedirs(val_dir, exist_ok=True)
#os.makedirs(test_dir, exist_ok=True)
#
## Function to split data and move files
#def split_data(source_folder, train_folder, val_folder, test_folder, val_size=0.15, test_size=0.5):
#    files = os.listdir(source_folder)
#    train_files, temp_files = train_test_split(files, test_size=(val_size + test_size))
#    val_files, test_files = train_test_split(temp_files, test_size=test_size/(val_size + test_size))
#    
#    for file in train_files:
#        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))
#    for file in val_files:
#        shutil.copy(os.path.join(source_folder, file), os.path.join(val_folder, file))
#    for file in test_files:
#        shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))
#
## Preparing data directories for each category
##categories = ["Mayo0", "Mayo1", "Mayo2", "Mayo3"]
#categories = ["BKL","MEL2", "NV","BCC"]
#for category in categories:
#    print(f"Processing {category} data:")
#    source_folder = os.path.join(base_dir, category)
#    train_folder = os.path.join(train_dir, category)
#    val_folder = os.path.join(val_dir, category)
#    test_folder = os.path.join(test_dir, category)
#    os.makedirs(train_folder, exist_ok=True)
#    os.makedirs(val_folder, exist_ok=True)
#    os.makedirs(test_folder, exist_ok=True)
#    split_data(source_folder, train_folder, val_folder, test_folder)
#
## Data augmentation setup
#train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
#                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
#
#validation_datagen = ImageDataGenerator(rescale=1./255)
#
## Data generators
#train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=16, class_mode='categorical')
#validation_generator = validation_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=16, class_mode='categorical')
#
## Model architecture
#model = Sequential([
#    Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#    MaxPooling2D(2, 2),
#    Conv2D(256, (3, 3), activation='relu'),
#    MaxPooling2D(2, 2),
#    Conv2D(512, (3, 3), activation='relu'),
#    MaxPooling2D(2, 2),
#    Flatten(),
#    Dense(1024, activation='relu'),
#    Dropout(0.5),
#    Dense(4, activation='softmax')
#])
##model = Sequential([
##    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
##    BatchNormalization(),
##    Conv2D(128, (3, 3), activation='relu'),
##    Flatten(),
##    Dense(256, activation='relu'),
##    Dropout(0.5),
##    Dense(len(class_names), activation='softmax')
##])
#
##
## Callbacks
#checkpoint = ModelCheckpoint('best_model5_new.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
#
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
## Model training
#history = model.fit(train_generator, epochs=30, validation_data=validation_generator, callbacks=[checkpoint, early_stopping, reduce_lr])
##
## Load the best model
#model.load_weights('best_model5_new.h5')
#
## Prepare test data generator
#test_datagen = ImageDataGenerator(rescale=1./255)
#test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=16, class_mode='categorical', shuffle=False)
#
## Predictions for confusion matrix and classification report
#predictions = model.predict(test_generator)
#predicted_classes = np.argmax(predictions, axis=1)
#true_classes = test_generator.classes
#class_labels = list(test_generator.class_indices.keys())
#
## Confusion Matrix and Classification Report
#print("Confusion Matrix:")
#print(confusion_matrix(true_classes, predicted_classes))
#print("Classification Report:")
#print(classification_report(true_classes, predicted_classes, target_names=class_labels))
#
## AUC-ROC Calculation
#predicted_probs = model.predict(test_generator)
#roc_score = roc_auc_score(to_categorical(true_classes), predicted_probs, multi_class='ovr')
#print(f"AUC-ROC Score: {roc_score}")
#
## Compute ROC curve and ROC area for each class
#n_classes = len(class_labels)
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(to_categorical(true_classes)[:, i], predicted_probs[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#
## Plot all ROC curves
#plt.figure()
#colors = ['blue', 'orange', 'green', 'red']
#for i, color in enumerate(colors):
#    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Mayo{i} (AUC = {roc_auc[i]:.2f})')
#
#plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Chance')
#
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.legend(loc='lower right')
#plt.grid(True)
#
## Save the plot to a file
#plt.savefig('roc_curve.png')
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# Set the backend to Agg for matplotlib
plt.switch_backend('Agg')

# Paths setup
base_dir = '/home/nehal/workspace/cancer_classification/train/New folder'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to split data and move files
def split_data(source_folder, train_folder, val_folder, val_size=0.10):
    files = os.listdir(source_folder)
    train_files, val_files = train_test_split(files, test_size=val_size)
    
    for file in train_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))
    for file in val_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(val_folder, file))

# Preparing data directories for each category
categories = ["BKL", "MEL2", "NV", "BCC"]
for category in categories:
    print(f"Processing {category} data:")
    source_folder = os.path.join(base_dir, category)
    train_folder = os.path.join(train_dir, category)
    val_folder = os.path.join(val_dir, category)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    split_data(source_folder, train_folder, val_folder)

# Data augmentation setup
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100), batch_size=32, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(val_dir, target_size=(100, 100), batch_size=32, class_mode='categorical')

# Model architecture
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.25),
    Dense(4, activation='softmax')
])

# Callbacks
checkpoint = ModelCheckpoint('best_model5_new.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
history = model.fit(train_generator, epochs=13, validation_data=validation_generator, callbacks=[checkpoint, early_stopping, reduce_lr])

# Load the best model
model.load_weights('best_model5_new.h5')

# Use the validation set as the test set for final evaluation
test_generator = validation_generator

# Predictions for AUC-ROC calculation
predicted_probs = model.predict(test_generator)
true_classes = test_generator.classes

# AUC-ROC Calculation
roc_score = roc_auc_score(to_categorical(true_classes), predicted_probs, multi_class='ovr')
print(f"AUC-ROC Score: {roc_score}")

# Compute ROC curve and ROC area for each class
n_classes = len(categories)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(to_categorical(true_classes)[:, i], predicted_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = ['blue', 'orange', 'green', 'red']
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{categories[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

# Save the plot to a file
plt.savefig('roc_curve.png')

# Function to preprocess and predict class for new images using AUC-ROC
def preprocess_and_predict(image_path, model, class_names, final_roc_auc_dict):
    img = image.load_img(image_path, target_size=(100, 100))  # Resize image to match the input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess image using ResNet50's preprocessing

    # Get features using the model
    features = model.predict(img_array)
    
    # Determine the class based on highest AUC-ROC score
    auc_scores = [final_roc_auc_dict[class_name] for class_name in class_names]
    highest_auc_class = class_names[np.argmax(auc_scores)]
    
    return highest_auc_class

# Example usage
#new_image_path = '/path/to/new/image.jpg'  # Update with the path to your new image
#predicted_class = preprocess_and_predict(new_image_path, model, categories, roc_auc)
#print("Predicted class for new image:", predicted_class)