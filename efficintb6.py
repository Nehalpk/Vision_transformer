import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.applications import EfficientNetB6
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import tensorflow as tf

warnings.filterwarnings("ignore", message="TF-TRT Warning: Could not find TensorRT")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setting GPU memory growth if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True) 

# Constants
data_dir = '/home/nehal/workspace/cancer_classification/train/data' 
batch_size = 8
img_height, img_width = 150, 150  # Custom input size for better detail
epochs = 50  # Maximum epochs (early stopping will stop earlier if needed)
num_classes = 4  # BKL, MEL2, NV, BCC

with tf.device('/device:GPU:0'):
    # Loading the datasets
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Get class names
    class_names = train_ds.class_names
    print(f"Class names: {class_names}")

    # Building the model for 4-class classification
    base_model = EfficientNetB6(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Changed to 4 classes with softmax
    ])

    # Compiling the model
    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # For integer labels
        metrics=['accuracy']
    )

    # Print model summary
    print("\nModel Summary:")
    model.summary()

    # Setup callbacks
    callbacks = [
        # Stop training when validation loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True,
            mode='min'
        ),
        
        # Save the best model during training
        ModelCheckpoint(
            filepath='best_efficientnetb6_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7,
            mode='min'
        ),
        
        # Log training history to CSV
        CSVLogger('training_log.csv', separator=',', append=False)
    ]
    
    print("\nCallbacks configured:")
    print("  ✓ EarlyStopping: patience=10, monitor=val_loss")
    print("  ✓ ModelCheckpoint: saves to 'best_efficientnetb6_model.h5'")
    print("  ✓ ReduceLROnPlateau: patience=5, factor=0.5")
    print("  ✓ CSVLogger: saves to 'training_log.csv'")

    # Training the model
    print("\nStarting training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # Plotting training accuracy and loss
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train', marker='o')
    plt.plot(history.history['val_accuracy'], label='validation', marker='s')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train', marker='o')
    plt.plot(history.history['val_loss'], label='validation', marker='s')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Predicting on validation set
    print("\nGenerating predictions on validation set...")
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Get class with highest probability
    
    # Get true labels
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    
    # Print classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print per-class accuracy
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY")
    print("="*80)
    for i, class_name in enumerate(class_names):
        class_accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"{class_name:>10}: {class_accuracy:>7.2%} ({cm[i, i]}/{cm[i].sum()} correct)")
    
    # Overall accuracy
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print(f"\n{'Overall':>10}: {overall_accuracy:>7.2%} ({np.trace(cm)}/{np.sum(cm)} correct)")
    print("="*80)
    
    # Note about saved model
    print(f"\n✓ Best model automatically saved to: 'best_efficientnetb6_model.h5'")
    print(f"✓ Training history saved to: 'training_log.csv'")
    print(f"✓ Training completed in {len(history.history['loss'])} epochs (max: {epochs})")
    
    # Sample predictions with confidence
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (First 10)")
    print("="*80)
    print(f"{'True Label':<15} {'Predicted':<15} {'Confidence':<12} {'Correct':<10}")
    print("-"*80)
    for i in range(min(10, len(y_true))):
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        confidence = y_pred_probs[i][y_pred[i]]
        correct = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"{true_label:<15} {pred_label:<15} {confidence:<12.2%} {correct:<10}")
    print("="*80)