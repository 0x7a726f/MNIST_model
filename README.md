## Model 1 Results

Loss: `0.026705551892518997`

Accuracy: `0.9939000606536865`

## Model 1 Configuration

```python
model = (
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
    BatchNormalization()
    MaxPooling2D((2, 2))
    Conv2D(64, (3, 3), activation='relu')
    BatchNormalization()
    MaxPooling2D((2, 2))
    Conv2D(64, (3, 3), activation='relu')
    BatchNormalization()
    Flatten()
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    Dropout(0.5)
    Dense(10, activation='softmax')
)
```
## Model 2 Results

Loss: `0.05337928980588913`

Accuracy: `0.9969000492648365`

## Model 2 Configuration

```python
model = (
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))
    BatchNormalization()
    MaxPooling2D((2, 2))
    Conv2D(128, (3, 3), activation='relu')
    BatchNormalization()
    Conv2D(128, (3, 3), activation='relu')
    BatchNormalization()
    Conv2D(256, (3, 3), activation='relu')
    BatchNormalization()
    MaxPooling2D((2, 2))
    Flatten()
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    Dropout(0.5)
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    Dropout(0.4)
    Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    Dropout(0.4)
    Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    Dropout(0.5)
    Dense(10, activation='softmax')
)
```
