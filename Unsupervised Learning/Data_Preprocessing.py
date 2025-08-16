from PIL import Image
import numpy as np
import io
import pandas as pd
import os


main_dir = os.path.dirname(__file__)

# Image data for PCA
data_path = os.path.join(main_dir, 'test-00000-of-00001.parquet')
data = pd.read_parquet(data_path, engine='pyarrow')

# Data for K-means
lloyd_data_path = os.path.join(main_dir, 'cm_dataset_2.csv')
lloyd_data = pd.read_csv(lloyd_data_path, names=['x1', 'x2'])
array_lloyd = lloyd_data.to_numpy()

# fixing randomness for controled experiments
np.random.seed(400)

# Sampleing 100 images for each label
indicies = []
for i in range(10):
    indicies.append(np.random.choice(data.index[data.label == i], size=100, replace=False))


image_data = np.empty(shape=(1, 784))
label_data = []
for i in range(10):
    for j in range(100):
        label_data.append(i)
        image = Image.open(io.BytesIO(data.image[indicies[i][j]]['bytes'])).convert('L')
        image_resized = image.resize((28, 28), Image.LANCZOS)
        image_array = np.expand_dims(np.array(image_resized, dtype=np.float32).flatten(), 0)
        image_data = np.concatenate((image_data, image_array), axis=0)
image_data = image_data[1:]

