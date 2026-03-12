import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import numpy as np
def loadnpredict():
    model=load_model(path)
    files = [f for f in os.listdir(test) if f.endswith('.npy')]
    if not files:
        print("No .npy files found in the directory.")
        return
    print(f"Found {len(files)} files. Starting inference...\n")
    print(f"{'File Name':<30} | {'Raw Score':<12} | {'Prediction'}")
    print("-" * 60)
    for filename in files:
        filepath=os.path.join(test,filename)
        data=np.load(filepath)
      def batch_predict():
    if not os.path.exists(path):
        print("Model file not found.")
        return    
    model = load_model(path)
    files = [f for f in os.listdir(test) if f.endswith('.npy')]
    for file_name in files :
        data = np.load(os.path.join(test, file_name))
    
        input_data = data[..., np.newaxis] 
        print(np.shape(input_data))
        predictions=model.predict(input_data,verbose=0)
        binary_preds = (predictions > threshold).astype(int)
        abnormal_count = np.sum(binary_preds)
        print(f"--- File: {file_name} ---")
        print(f"Overall File Status: {'ABNORMAL' if abnormal_count > 0 else 'NORMAL'}\n")
      if __name__ == "__main__":
    batch_predict()
