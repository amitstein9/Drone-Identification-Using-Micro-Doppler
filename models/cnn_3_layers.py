import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import tensorflow as tf      

from tensorflow.keras import layers, applications
from sklearn.metrics import classification_report
# at top of file
from plot_utils import plot_accuracy_curve, plot_loss_curve, plot_confusion_matrix


# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=16,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=6,
    min_lr=1e-6,
    verbose=1
)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="training/checkpoints/cnn/best.h5",
    save_weights_only=True,
    save_freq='epoch',
    verbose=1,
    monitor='val_accuracy',
    save_best_only=True
)


# [I 2025-04-21 08:44:45,093] Trial 16 finished with value: 0.8670998811721802 and parameters: {'lr': 0.00022143712026018205, 'dropout': 0.3442070058499148, 'l2_reg': 0.009787704208545057, 'batch_size': 16, 'patience': 16, 'f1': 64, 'f2': 192}. Best is trial 16 with value: 0.8670998811721802.

DROPOUT = 0.3
F1, F2, F3 = 32, 64, 128  # Filter sizes for the 3 conv layers
LR = 1e-3  # Learning rate

def create_cnn(input_shape, num_classes):
    """
    Creates and compiles the CNN model with:
     - 3 conv blocks (MaxPool)
     - 1 fully-connected Dense head
    """
    model = keras.Sequential([ 
        layers.Input(shape=input_shape),
        
        # Conv block 1
        layers.Conv2D(F1, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(LR)),
        layers.MaxPooling2D((2, 2)),  # 2x2 MaxPooling

        # Conv block 2
        layers.Conv2D(F2, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(LR)),
        layers.MaxPooling2D((2, 2)),  # 2x2 MaxPooling
        
        # Conv block 3
        layers.Conv2D(F3, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(LR)),
        layers.MaxPooling2D((2, 2)),  # 2x2 MaxPooling

        layers.Flatten(),  # Flatten the output
        
        # Fully connected layer
        layers.Dense(128, activation='relu', 
                     kernel_regularizer=regularizers.l2(LR)),
        layers.Dropout(DROPOUT),  # Dropout after fully connected layer

        # Final softmax layer for classification
        layers.Dense(num_classes, activation='softmax')
    ])

    
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def evaluate_and_save_results(history, model_type, model, X_val, y_val, results_dir):
    # 1) Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"\nValidation set accuracy: {val_accuracy:.4f}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results in: {results_dir}")

    # 2) Load previous history if it exists
    hist_file = os.path.join(results_dir, 'history.npz')
    if os.path.exists(hist_file):
        prev = np.load(hist_file)
        acc_prev      = prev['acc'].tolist()
        val_acc_prev  = prev['val_acc'].tolist()
        loss_prev     = prev['loss'].tolist()
        val_loss_prev = prev['val_loss'].tolist()
    else:
        acc_prev = val_acc_prev = loss_prev = val_loss_prev = []

    # 3) Append new run’s history
    acc_new      = history.history['accuracy']
    val_acc_new  = history.history['val_accuracy']
    loss_new     = history.history['loss']
    val_loss_new = history.history['val_loss']

    acc_combined      = acc_prev      + acc_new
    val_acc_combined  = val_acc_prev  + val_acc_new
    loss_combined     = loss_prev     + loss_new
    val_loss_combined = val_loss_prev + val_loss_new

    # 4) Save the updated history back to disk
    np.savez(
        hist_file,
        acc=acc_combined,
        val_acc=val_acc_combined,
        loss=loss_combined,
        val_loss=val_loss_combined
    )

    # 5) Plot the cumulative accuracy & loss
    acc_path = os.path.join(results_dir, 'accuracy_plot.png')
    plot_accuracy_curve(
        acc_combined, val_acc_combined, acc_path,
        title=f"{model_type} Accuracy (Final Val Acc: {val_acc_combined[-1]:.4f})"
    )
    print(f"Accuracy plot saved to: {acc_path}")

    loss_path = os.path.join(results_dir, 'loss_plot.png')
    plot_loss_curve(
        loss_combined, val_loss_combined, loss_path,
        title=f"{model_type} Loss"
    )
    print(f"Loss plot saved to: {loss_path}")

    # 6) Save model
    final_model_path = os.path.join(results_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # 7) Confusion matrix & classification report
    y_pred = np.argmax(model.predict(X_val), axis=1)
    labels = sorted(np.unique(y_val).tolist())
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_val, y_pred, labels, cm_path, normalize=True)
    print(f"Confusion matrix saved to: {cm_path}")


    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))