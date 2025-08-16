import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from principal_component_analysis import PCA
from k_means import Llyoid_Algorithm
from Data_Preprocessing import image_data, label_data, lloyd_data, array_lloyd
from scipy.spatial import Voronoi, voronoi_plot_2d


# fixing randomness for controlled experimentation
np.random.seed(400)

# 1st Qusetion
# (i)
components = 10
pca_algorithm = PCA(components=components)
pca_algorithm.fit(image_data)


all_components = pca_algorithm.get_eigenvectors()
all_eigenValues = pca_algorithm.get_eigenvalues()
sum_eigenVlaues = np.sum(all_eigenValues)
principal_components = all_components[:, :components]
data_mean = pca_algorithm.get_mean()

# Code for Visualizing Principal Components
def component_visualization(components, sum_values, eigenvalues, number_columns=5, image_shape=(28, 28)):
    number_of_images = components.shape[1]
    # Set up the grid for plotting
    rows = (number_of_images + number_columns - 1) // number_columns
    fig, axes = plt.subplots(rows, number_columns, figsize=(10, 2 * rows))

    for i in range(number_of_images):
        image = components[:, i].reshape(image_shape)
        variance_explained = eigenvalues[i]/sum_values
        ax = axes[i // number_columns, i % number_columns] if rows > 1 else axes[i % number_columns]
        ax.imshow(image, cmap="gray")
        ax.set_title(f'Variance = {round(variance_explained*100,2)}')
        ax.axis("off")
    plt.suptitle('Visualizing Principal Components')
    plt.tight_layout()
    plt.show()


# Picking random image from sampled dataset to visualize images of the principal component
component_visualization(components=principal_components, sum_values=sum_eigenVlaues, eigenvalues=all_eigenValues)


# (ii)
# Reconstructing Data

def reconstructed_images(image_data, labels, number_components, num_images=10, number_columns=5, image_shape=(28, 28)):
    # Number of columns in the grid
    rows = (num_images + number_columns - 1) // number_columns  # Calculate rows dynamically
    fig, axes = plt.subplots(rows, number_columns, figsize=(10, 2 * rows))

    for i in range(rows * number_columns):
        ax = axes[i // number_columns, i % number_columns] if rows > 1 else axes[i % number_columns]
        if i < num_images:
            # Reshape flattened array to original image shape
            index = np.random.randint(0, 1000)
            image = image_data[index].reshape(image_shape)
            ax.set_title(f'label: {labels[index]}')
            ax.imshow(image, cmap="gray")
            ax.axis("off")
        else:
            ax.axis("off")
    plt.suptitle(f'Reconstructed Data using {number_components} number of components')
    plt.tight_layout()
    plt.show()


for i in range(50, 350, 50):
    components = 10
    pca_algorithm = PCA(components=i)
    pca_algorithm.fit(image_data)
    reconstructed_images(pca_algorithm.reconstruct(), labels=label_data, number_components=i)



num_components = np.arange(len(all_eigenValues))
variance_pc = []
prev_eigenvalue = 0
dimension = None
for i in range(len(all_eigenValues)):
    prev_eigenvalue += all_eigenValues[i]
    varaince_exp = prev_eigenvalue/sum_eigenVlaues
    if not dimension and varaince_exp > 0.95:
        dimension = i
    variance_pc.append(varaince_exp)
plt.plot(num_components, variance_pc, 'b*')
plt.title('Variance In Data Explained As Number Of Components Increases')
plt.xlabel('Number of Componentes')
plt.ylabel('Varaince In Data Explained')
plt.show()
print(f'I will pick dimension d = {dimension} as at this dimension variance expalined by components are > 95%')

# 2nd Question
# (i) 5 different random initialization and plot the error function w.r.t iterations in each case
random_initials = [400, 300, 1221, 1044, 1001]
for i in range(len(random_initials)):
    model = Llyoid_Algorithm(number_of_clustors=2)
    cluster_label = model.fit(array_lloyd, random_state=random_initials[i], iterations=10)
    error = model.get_error()
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(10), error)
    plt.title(f'Error Vs Iterations for {i+1} Random Initialization')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Error')
    plt.subplot(1, 2, 2)
    plt.scatter(x = lloyd_data.x1, y = lloyd_data.x2, c=cluster_label)
    plt.title(f'Clusters for {i} Random Initialization')
    plt.tight_layout()
    plt.show()

# (ii)
def plot_voronoi(centroids, data):
    if len(centroids) > 3:
        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2)
        plt.scatter(data[:, 0], data[:, 1], c="blue", s=1)
        plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x")
        plt.show()
    else:
        x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
        y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        distances = np.linalg.norm(mesh_points[:, np.newaxis] - means, axis=2)
        labels = np.argmin(distances, axis=1)
        labels = labels.reshape(xx.shape)
        plt.contour(xx, yy, labels, alpha=0.3, cmap='viridis')
        plt.scatter(data[:, 0], data[:, 1], c='black', alpha=0.7, s=10)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=200, linewidths=3)
        plt.title(f'Voroni Region for k={len(centroids)}')
        plt.show()


for k in range(2, 6):
    model = Llyoid_Algorithm(number_of_clustors=k)
    cluster_label = model.fit(array_lloyd)
    means = model.get_mean()
    plot_voronoi(centroids=means, data=array_lloyd)
