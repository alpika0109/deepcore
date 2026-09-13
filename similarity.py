# Section 5: Similarity Calculation Using the Siamese Network
# Extract features for two clothing items to compare similarity
features1 = extract_features("dataset/clothes/clothing_1.jpg")
features2 = extract_features("dataset/clothes/clothing_2.jpg")

# Predict similarity using the Siamese Network
similarity = siamese_model.predict([features1.reshape(1, -1), features2.reshape(1, -1)])
print(f"AI Model Prediction: {similarity[0][0]:.2f}")

if similarity[0][0] > 0.5:
    print("Clothes are similar!")
else:
    print("Clothes are NOT similar.")
# Section 6: Evaluating and Testing the Model
# You can test and evaluate the trained model on a test dataset.
test_loss, test_acc = siamese_model.evaluate([test_images[0], test_images[1]], [0], verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")
