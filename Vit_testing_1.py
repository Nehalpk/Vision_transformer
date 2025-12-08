import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from project import ViT
from glob import glob
from sklearn.utils import shuffle
from project import ClassToken
import cv2
from patchify import patchify
hp = {
    "image_size": 512,
    "num_channels": 3,
    "patch_size": 64,
    "num_classes": 4,
    "class_names": ["BKL","MEL2", "NV","BCC"],
    "num_layers": 12,
    "hidden_dim": 768,
    "mlp_dim": 3072,
    "num_heads": 4,
    "dropout_rate": 0.1,
    "batch_size": 8,
    "lr": 1e-4,
    "num_epochs": 2
}
hp["num_patches"] = (hp["image_size"] ** 2) // (hp["patch_size"] ** 2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"] * hp["patch_size"] * hp["num_channels"])


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path, split=0.20):
    images = glob(os.path.join(path, "*", "*.jpg"))

    # Separate images by class
    images_class_BKL = [image for image in images if "BKL" in image]
    images_class_NV = [image for image in images if "NV" in image]
    images_class_MEL2 = [image for image in images if "MEL2" in image]
    #images_class_MEL2 = [image for image in images if "MEL2" in image]
    images_class_BCC = [image for image in images if "BCC" in image]
    

    # Oversample to 33,000 to match the largest class (BKL)
    target_size = len(images_class_BKL)
    
    images_class_NV = np.random.choice(images_class_NV, size=target_size, replace=True).tolist()
    images_class_MEL2 = np.random.choice(images_class_MEL2, size=target_size, replace=True).tolist()
    images_class_BCC = np.random.choice(images_class_BCC, size=target_size, replace=True).tolist()
    # Concatenate lists again and shuffle
    images = shuffle(images_class_BKL + images_class_NV + images_class_MEL2 + images_class_BCC)

    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    return train_x, valid_x, test_x


def process_image_label(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    image = image / 255.0

    patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
    patches = patchify(image, patch_shape, hp["patch_size"])
    patches = np.reshape(patches, hp["flat_patches_shape"])
    patches = patches.astype(np.float32)

    class_name = path.split("/")[-2]
    class_idx = hp["class_names"].index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)

    return patches, class_idx


def parse(path):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, hp["num_classes"])

    patches.set_shape(hp["flat_patches_shape"])
    labels.set_shape(hp["num_classes"])

    return patches, labels


def tf_dataset(images, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8)
    return ds

# The main execution part
if __name__ == "__main__":
    dataset_path = '/home/nehal/workspace/cancer_classification/train/data'
    _, _, test_x = load_data(dataset_path)
    
    # Evaluating on the test set
    print("Evaluating the test set...")
    test_ds = tf_dataset(test_x, batch=hp["batch_size"])
    all_preds = []

    for fold_no in range(1, 6):
        fold_model = tf.keras.models.load_model(f"files/modelN_fold{fold_no}.h5", custom_objects={'ClassToken': ClassToken})
 #fold_model = tf.keras.models.load_model(f"files/modelN_fold{fold_no}.h5")
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

def sample_images_per_category(images, num_samples=2):
    sampled_images = {}
    for class_name in hp["class_names"]:
        class_images = [img for img in images if class_name in img]
        sampled_images[class_name] = np.random.choice(class_images, size=num_samples, replace=False).tolist()
    return sampled_images

sampled = sample_images_per_category(test_x)

for class_name, img_paths in sampled.items():
    fig, axes = plt.subplots(1, len(img_paths), figsize=(10, 5))
    
    for ax, img_path in zip(axes, img_paths):
        # Predict for the current image
        img_ds = tf_dataset([img_path], batch=1)
        probs = fold_model.predict(img_ds)[0]  # Taking the prediction of the first fold model as an example
        
        # Load and display the image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        
        # Setting title with the predicted probabilities
        title = "\n".join([f"{cls}: {prob:.2f}" for cls, prob in zip(hp["class_names"], probs)])
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
