from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from project import ViT
import numpy as np
import tensorflow as tf
import os
from main import load_data, tf_dataset
from glob import glob
from sklearn.utils import shuffle

hp = {}
hp["image_size"] = 512
hp["num_channels"] = 3
hp["patch_size"] = 64
hp["num_patches"] = (hp["image_size"] ** 2) // (hp["patch_size"] ** 2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"] * hp["patch_size"] * hp["num_channels"])

hp["batch_size"] = 4
hp["lr"] =1e-4
hp["num_epochs"] = 1  
hp["num_classes"] = 2
hp["class_names"] = ["BKL","MEL2"]

hp["num_layers"] = 4
hp["hidden_dim"] = 768
hp["mlp_dim"] =3072
hp["num_heads"] = 4
hp["dropout_rate"] = 0.1


def load_test_data(path):
    images = shuffle(glob(os.path.join(path, "*", "*.jpg")))
    return images

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Paths """
    testset_path = '/home/nehal/workspace/cancer_classification/output_images/train'
    model_path = os.path.join("files", "model3_fold3.h5")

    """ Dataset """
    test_x = load_test_data(testset_path)
    print(f"Test: {len(test_x)}")
    test_ds = tf_dataset(test_x, batch=hp["batch_size"])

    """ Model """
    model = ViT(hp)
    model.load_weights(model_path)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(hp["lr"]),
        metrics=["acc"]
    )
    model.evaluate(test_ds)
    

    y_true = []
    y_pred = []
    y_pred_probs_list = []

    for i, (x, y) in enumerate(test_ds):
        y_pred_probs = model.predict(x)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        y_pred.extend(y_pred_labels)
        y_true_labels = np.argmax(y, axis=1)
        y_true.extend(y_true_labels)
        y_pred_probs_list.extend(y_pred_probs)
        for j in range(len(y_pred_labels)):
            if y_pred_labels[j] != y_true_labels[j]:
                print(f"Wrong prediction for image: {test_x[i*hp['batch_size']+j]}, predicted class: {hp['class_names'][y_pred_labels[j]]}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_onehot = to_categorical(y_true)
    y_pred_onehot = to_categorical(y_pred)
    y_pred_probs_array = np.array(y_pred_probs_list)

    print(f"Classification Report: \n{classification_report(y_true, y_pred, target_names=hp['class_names'])}")

    roc_auc = roc_auc_score(y_true_onehot, y_pred_probs_array, multi_class='ovr')
    print(f"AUC-ROC: {roc_auc}")

    # ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true_bin = lb.transform(y_true)

    for i in range(hp["num_classes"]):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs_array[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_pred_probs_array[:, i])

    # Plot of a ROC curve for a specific class
    for i in range(hp["num_classes"]):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for %s' % (roc_auc[i], hp["class_names"][i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
