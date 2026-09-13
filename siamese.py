# Section 4: Siamese Network for Clothing Similarity Detection
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Siamese Network definition
def build_siamese_model(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3,3), activation='relu')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    return Model(input, x)

# Create twin models
input_shape = (224, 224, 3)  # Since we are using ResNet50 features (224x224x3)
base_model = build_siamese_model(input_shape)

inputA = Input(shape=input_shape)
inputB = Input(shape=input_shape)

featA = base_model(inputA)
featB = base_model(inputB)

# Compute Euclidean distance between feature vectors
distance = Lambda(lambda x: tf.abs(x[0] - x[1]))([featA, featB])
output = Dense(1, activation='sigmoid')(distance)

siamese_model = Model(inputs=[inputA, inputB], outputs=output)
siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Example: Training data (replace with actual data and labels)
# We are assuming that 'pairs' and 'labels' are available here
pairs = [features1, features2]  # Should be a list of image pairs
labels = [1]  # Corresponding binary labels (1 for similar, 0 for dissimilar)

# Train the model
siamese_model.fit([pairs[0], pairs[1]], labels, epochs=10)
