# yelp_sentiment_dnn_gpu.py
# Deep Feedforward Neural Network for Yelp Review Sentiment Analysis
# Optimized for GPU execution in VS Code

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time
import os

# GPU Configuration and Verification
print("=" * 60)
print("GPU SETUP AND VERIFICATION")
print("=" * 60)

# Check TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs Available: {len(gpus)}")

if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
        
        # Print GPU details
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            
        # Set GPU as default device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
        
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("WARNING: No GPU found. Running on CPU.")

# Verify GPU is being used
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    print(f"Test tensor device: {test_tensor.device}")

print("\n" + "=" * 60)
print("DATA LOADING AND PREPROCESSING")
print("=" * 60)

# Load Yelp Polarity Reviews dataset
print("Loading Yelp Polarity Reviews dataset...")
try:
    (train_data, test_data), ds_info = tfds.load(
        'yelp_polarity_reviews',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True,
        download=True
    )
    print("Dataset loaded successfully!")
    print(f"Training samples: {ds_info.splits['train'].num_examples}")
    print(f"Test samples: {ds_info.splits['test'].num_examples}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Configuration parameters
VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 250
BATCH_SIZE = 64
EPOCHS = 10

# Text vectorization
print("Setting up text vectorization...")
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

# Adapt vectorizer to training text
train_text = train_data.map(lambda text, label: text)
vectorizer.adapt(train_text)
print("Text vectorization setup complete!")

# Prepare datasets
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorizer(text), label

print("Preparing training and test datasets...")
train_ds = train_data.map(vectorize_text).cache().shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_data.map(vectorize_text).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)

# Build Deep Feedforward Neural Network (DNN) with only Dense layers
def create_dnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(SEQUENCE_LENGTH,)),
        tf.keras.layers.Dense(512, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', name='dense_3'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu', name='dense_4'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    return model

# Force model creation on GPU
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = create_dnn_model()

model.summary()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
]

# Train model on GPU
print("Starting training...")
start_time = time.time()

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Evaluate model
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_ds, verbose=1)
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print(f"\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {f1_score:.4f}")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Precision & Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.show()

# Generate predictions for confusion matrix
print("\nGenerating predictions for detailed analysis...")
y_true = []
y_pred = []

for x_batch, y_batch in test_ds:
    predictions = model.predict(x_batch, verbose=0)
    y_true.extend(y_batch.numpy())
    y_pred.extend((predictions > 0.5).astype(int).flatten())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

print("\n" + "=" * 60)
print("EXPERIMENT VARIATIONS")
print("=" * 60)

# Experiment 1: Different Architecture
def create_smaller_dnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(SEQUENCE_LENGTH,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Experiment 2: Different Architecture with more layers
def create_deeper_dnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(SEQUENCE_LENGTH,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Function to train and evaluate different models
def train_and_evaluate_model(model_func, model_name, epochs=5):
    print(f"\nTraining {model_name}...")
    
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model = model_func()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    start_time = time.time()
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs,
            verbose=0
        )
    
    training_time = time.time() - start_time
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    print(f"{model_name} Results:")
    print(f"  Training Time: {training_time:.2f} seconds")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    
    return history, test_acc, test_loss

# Run experiments
experiments = [
    (create_smaller_dnn, "Smaller DNN (2 hidden layers)"),
    (create_deeper_dnn, "Deeper DNN (5 hidden layers)")
]

experiment_results = {}
for model_func, model_name in experiments:
    history, acc, loss = train_and_evaluate_model(model_func, model_name)
    experiment_results[model_name] = {'accuracy': acc, 'loss': loss, 'history': history}

print("\n" + "=" * 60)
print("OPTIMIZER COMPARISON")
print("=" * 60)

# Test different optimizers
optimizers = [
    (tf.keras.optimizers.SGD(learning_rate=0.01), "SGD"),
    (tf.keras.optimizers.RMSprop(learning_rate=0.001), "RMSprop"),
    (tf.keras.optimizers.Adam(learning_rate=0.001), "Adam")
]

optimizer_results = {}
for optimizer, opt_name in optimizers:
    print(f"\nTesting {opt_name} optimizer...")
    
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model = create_dnn_model()
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    start_time = time.time()
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=3,  # Reduced epochs for comparison
            verbose=0
        )
    
    training_time = time.time() - start_time
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    optimizer_results[opt_name] = {
        'accuracy': test_acc,
        'loss': test_loss,
        'time': training_time
    }
    
    print(f"{opt_name} Results:")
    print(f"  Training Time: {training_time:.2f} seconds")
    print(f"  Test Accuracy: {test_acc:.4f}")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"GPU Used: {'Yes' if gpus else 'No'}")
print(f"Dataset: Yelp Polarity Reviews")
print(f"Training Samples: {ds_info.splits['train'].num_examples}")
print(f"Test Samples: {ds_info.splits['test'].num_examples}")
print(f"Vocabulary Size: {VOCAB_SIZE}")
print(f"Sequence Length: {SEQUENCE_LENGTH}")

print("\nOptimizer Comparison Results:")
for opt_name, results in optimizer_results.items():
    print(f"  {opt_name}: Accuracy={results['accuracy']:.4f}, Time={results['time']:.2f}s")

print("\nArchitecture Comparison Results:")
for model_name, results in experiment_results.items():
    print(f"  {model_name}: Accuracy={results['accuracy']:.4f}")

print("\nExperiment completed successfully!")

# Save model
model_save_path = "yelp_sentiment_dnn_model.h5"
model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")
# Load and use the saved model in an application
print("\n" + "=" * 60)
print("USING SAVED MODEL FOR PREDICTION")
print("=" * 60)

# Load the model
loaded_model = tf.keras.models.load_model(model_save_path)

# Example usage: Predict sentiment for new reviews
sample_reviews = [
    "The food was absolutely wonderful, from preparation to presentation, very pleasing.",
    "Worst experience ever. The service was terrible and the food was bland."
]

# Vectorize the sample reviews
sample_reviews_tensor = tf.constant(sample_reviews)
sample_reviews_vectorized = vectorizer(sample_reviews_tensor)

# Predict
predictions = loaded_model.predict(sample_reviews_vectorized)
for review, pred in zip(sample_reviews, predictions):
    sentiment = "Positive" if pred[0] > 0.5 else "Negative"
    print(f"Review: \"{review}\"")
    print(f"Predicted Sentiment: {sentiment} (Score: {pred[0]:.4f})\n")