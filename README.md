Loss: 0.026705551892518997
Accuracy: 0.9939000606536865

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

optimizer = "adam"
