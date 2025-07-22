import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model = load_model('person_identifier_model.h5')
IMG_SIZE = (128, 128)
label_map = {0: 'Normal', 1: 'Abnormal'}
TEST_DIR = 'C:/Users/Admin/OneDrive/Desktop/MachineLearning/dataset/test'
test_images = [os.path.join(TEST_DIR, fname) 
               for fname in os.listdir(TEST_DIR) 
               if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
TEST_DIR = 'C:/Users/Admin/OneDrive/Desktop/MachineLearning/dataset/test'

# Get all image paths in test folder
test_images = [os.path.join(TEST_DIR, fname) 
               for fname in os.listdir(TEST_DIR) 
               if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
def predict_person(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = 1 if prediction > 0.5 else 0
    confidence = prediction if label == 1 else 1 - prediction
    return label_map[label], confidence

for img_path in test_images:
    predicted_person, conf = predict_person(img_path)
    print(f"{os.path.basename(img_path)} → Predicted: {predicted_person} (Confidence: {conf:.2f})")
