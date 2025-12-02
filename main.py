#import os
#import cv2
#import numpy as np
#import tensorflow as tf
#from glob import glob
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split, KFold
#from keras.metrics import AUC
#from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
#from patchify import patchify
## Assuming you have already imported or defined the ViT model
## ...
#from sklearn.metrics import classification_report, roc_auc_score, roc_curve
#import matplotlib.pyplot as plt
#from project import ViT
#hp = {
#    "image_size": 300,
#    "num_channels": 3,
#    "patch_size": 10,
#    "num_classes": 4,
#    #"class_names": ["BKL","MEL2", "NV",'BCC'],
#    "class_names":["Mayo0","Mayo1","Mayo2","Mayo3"],
#    "num_layers": 4,
#    "hidden_dim": 768,
#    "mlp_dim": 3072,
#    "num_heads": 4,
#    "dropout_rate": 0.1,
#    "batch_size": 8,
#    "lr": 1e-4,
#    "num_epochs": 2
#}
#hp["num_patches"] = (hp["image_size"] ** 2) // (hp["patch_size"] ** 2)
#hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"] * hp["patch_size"] * hp["num_channels"])
#
#
#def create_dir(path):
#    if not os.path.exists(path):
#        os.makedirs(path)
#
#
#def load_data(path, split=0.20):
#    images = glob(os.path.join(path, "*", "*.jpg"))
#
#    # Separate images by class
#    images_class_BKL = [image for image in images if "Mayo0" in image]
#    images_class_NV = [image for image in images if "Mayo1" in image]
#    images_class_MEL2 = [image for image in images if "Mayo2" in image]
#    images_class_BCC = [image for image in images if "Mayo3" in image]
#
#    # Oversample to 33,000 to match the largest class (BKL)
#    target_size = len(images_class_BKL)
#    
#    images_class_NV = np.random.choice(images_class_NV, size=target_size, replace=True).tolist()
#    images_class_MEL2 = np.random.choice(images_class_MEL2, size=target_size, replace=True).tolist()
#    images_class_BCC = np.random.choice(images_class_BCC, size=target_size, replace=True).tolist()
#
#    # Concatenate lists again and shuffle
#    images = shuffle(images_class_BKL + images_class_NV + images_class_MEL2 + images_class_BCC)
#
#    split_size = int(len(images) * split)
#    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
#    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
#    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")
#
#    return train_x, valid_x, test_x
#
#
#def process_image_label(path):
#    path = path.decode()
#    image = cv2.imread(path, cv2.IMREAD_COLOR)
#    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
#    image = image / 255.0
#
#    patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
#    patches = patchify(image, patch_shape, hp["patch_size"])
#    patches = np.reshape(patches, hp["flat_patches_shape"])
#    patches = patches.astype(np.float32)
#
#    class_name = path.split("/")[-2]
#    class_idx = hp["class_names"].index(class_name)
#    class_idx = np.array(class_idx, dtype=np.int32)
#
#    return patches, class_idx
#
#
#def parse(path):
#    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
#    labels = tf.one_hot(labels, hp["num_classes"])
#
#    patches.set_shape(hp["flat_patches_shape"])
#    labels.set_shape(hp["num_classes"])
#
#    return patches, labels
#
#
#def tf_dataset(images, batch=32):
#    ds = tf.data.Dataset.from_tensor_slices((images))
#    ds = ds.map(parse).batch(batch).prefetch(8)
#    return ds
#
#
#def get_model():
#    model = ViT(hp)
#    model.compile(
#        loss="categorical_crossentropy",
#        optimizer=tf.keras.optimizers.Adam(hp["lr"], clipvalue=1.0),
#        #metrics=["acc", AUC(name='auc', multi_label=True)]
#        metrics=["acc", AUC(name='auc')]
#    )
#    return model
#
#
#if __name__ == "__main__":
#    create_dir("files")
#    #dataset_path = '/home/nehal/workspace/cancer_classification/train/data'
#    dataset_path = '/home/nehal/workspace/cancer_classification/New folder (4)'
#    model_path = os.path.join("files", "model3.h5")
#    csv_path = os.path.join("files", "log2.csv")
#    train_x, valid_x, test_x = load_data(dataset_path)
#    images = train_x + valid_x
#    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")
#    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#    fold_no = 5
#
#    for train, val in kfold.split(images):
#        print(f'Training for fold {fold_no} ...')
#        train_ds = tf_dataset(np.array(images)[train], batch=hp["batch_size"])
#        valid_ds = tf_dataset(np.array(images)[val], batch=hp["batch_size"])
#        model = get_model()
#        callbacks = [
#            ModelCheckpoint(f"files/modelN_fold{fold_no}.h5", monitor='val_loss', verbose=1, save_best_only=True),
#            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
#            CSVLogger(f"files/log2_fold{fold_no}.csv"),
#            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
#        ]
#        model.fit(train_ds, epochs=hp["num_epochs"], validation_data=valid_ds, callbacks=callbacks)
#        fold_no += 1
#    
#    # Evaluating on the test set
#    print("Evaluating the test set...")
#    test_ds = tf_dataset(test_x, batch=hp["batch_size"])
#    all_preds = []
#
#    for fold_no in range(1, 6):
#        fold_model = tf.keras.models.load_model(f"files/modelN_fold{fold_no}.h5")
#        y_pred_probs = fold_model.predict(test_ds)
#        all_preds.append(y_pred_probs)
#
#    avg_preds = np.mean(all_preds, axis=0)
#    final_preds = np.argmax(avg_preds, axis=1)
#    y_true = [hp["class_names"].index(img.split("/")[-2]) for img in test_x]
#    print(classification_report(y_true, final_preds, target_names=hp["class_names"]))
#    roc_aucs = []
#
#    for i, class_name in enumerate(hp["class_names"]):
#        auc_val = roc_auc_score((np.array(y_true) == i).astype(int), avg_preds[:, i])
#        roc_aucs.append(auc_val)
#        print(f"AUC-ROC ({class_name}): {auc_val:.4f}")
#
#    for i, class_name in enumerate(hp["class_names"]):
#        fpr, tpr, _ = roc_curve((np.array(y_true) == i).astype(int), avg_preds[:, i])
#        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_aucs[i]:.2f})")
#
#    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver Operating Characteristic (ROC) Curve')
#    plt.legend()
#    plt.show()
import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from keras.metrics import AUC
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from patchify import patchify
from keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from project import ViT, ClassToken

# Setup hyperparameters
hp = {
    "image_size": 150,
    "num_channels": 3,
    "patch_size": 10,
    "num_classes": 4,
    "class_names": ["BKL","MEL2", "NV","BCC"],
    "num_layers": 4,
    "hidden_dim": 512,
    "mlp_dim": 3072,
    "num_heads": 4,
    "dropout_rate": 0.1,
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 3,
}
#hp = {
#    "image_size": 300,
#    "num_channels": 3,
#    "patch_size": 10,
#    "num_classes": 4,
#    #"class_names": ["Mayo0", "Mayo1", "Mayo2", "Mayo3"],
#    "class_names": ["BKL","MEL2", "NV","BCC"],
#    "num_layers": 4,
#    "hidden_dim": 768,
#    "mlp_dim": 3072,
#    "num_heads": 4,
#    "dropout_rate": 0.1,
#    "batch_size": 8,
#    "lr": 1e-4,
#    "num_epochs": 3,
#}
hp["num_patches"] = (hp["image_size"] ** 2) // (hp["patch_size"] ** 2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"] * hp["patch_size"] * hp["num_channels"])

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.10):
    images = glob(os.path.join(path, "*", "*.jpg"))
    images_class = {name: [image for image in images if name in image] for name in hp["class_names"]}
    target_size = max(len(v) for v in images_class.values())
    
    for name in hp["class_names"]:
        images_class[name] = np.random.choice(images_class[name], size=target_size, replace=True).tolist()

    images = sum(images_class.values(), [])
    np.random.shuffle(images)

    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    #train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    test_x=valid_x
    return train_x, valid_x, test_x

def process_image_label(path):
    path = path.decode('utf-8')  # Ensure the path is decoded to string properly
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    image = image / 255.0

    patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
    patches = patchify(image, patch_shape, hp["patch_size"])
    patches = np.reshape(patches, hp["flat_patches_shape"]).astype(np.float32)

    class_name = path.split("/")[-2]
    class_idx = hp["class_names"].index(class_name)
    return patches, np.array(class_idx, dtype=np.int32)

def parse(path):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, hp["num_classes"])
    patches.set_shape(hp["flat_patches_shape"])
    labels.set_shape((hp["num_classes"],))
    return patches, labels

def tf_dataset(images, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8)
    return ds

def get_model():
    model = ViT(hp)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(hp["lr"]),
        metrics=["accuracy", AUC(name='auc')]
    )
    return model

if __name__ == "__main__":
    create_dir("files")
    dataset_path ='/home/nehal/workspace/cancer_classification/train/data' #'/home/nehal/workspace/cancer_classification/New folder'
    train_x, valid_x, test_x = load_data(dataset_path)

    kfold = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_no = 1

    for train, val in kfold.split(train_x + valid_x):
        print(f'Training for fold {fold_no} ...')
        train_ds = tf_dataset(np.array(train_x + valid_x)[train], batch=hp["batch_size"])
        valid_ds = tf_dataset(np.array(train_x + valid_x)[val], batch=hp["batch_size"])
        
        model = get_model()
        callbacks = [
            ModelCheckpoint(f"files/modelN_fold{fold_no}.h5", monitor='val_loss', verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
            CSVLogger(f"files/log2_fold{fold_no}.csv"),
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        ]
        model.fit(train_ds, epochs=hp["num_epochs"], validation_data=valid_ds, callbacks=callbacks)
        fold_no += 1

    # Evaluating on the test set
    print("Evaluating the test set...")
    test_ds = tf_dataset(test_x, batch=hp["batch_size"])
    all_preds = []

    custom_objects = {"ClassToken": ClassToken}  # This is required for loading the model with custom layers
    for fold_no in range(1, 4):
        fold_model = load_model(f"files/modelN_fold{fold_no}.h5", custom_objects=custom_objects)
        y_pred_probs = fold_model.predict(test_ds)
        all_preds.append(y_pred_probs)

    avg_preds = np.mean(all_preds, axis=0)
    final_preds = np.argmax(avg_preds, axis=1)
    y_true = [hp["class_names"].index(img.split("/")[-2]) for img in test_x]
    print(classification_report(y_true, final_preds, target_names=hp["class_names"]))
    roc_aucs = []

    for i, class_name in enumerate(hp["class_names"]):
        auc_val = roc_auc_score((np.array(y_true) == i).astype(int), avg_preds[:, i])
        roc_aucs.append(auc_val)
        print(f"AUC-ROC ({class_name}): {auc_val:.4f}")

    for i, class_name in enumerate(hp["class_names"]):
        fpr, tpr, _ = roc_curve((np.array(y_true) == i).astype(int), avg_preds[:, i])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_aucs[i]:.2f})")

    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
