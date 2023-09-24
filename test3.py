import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
 
# Generate synthetic data
def generate_synthetic_data(num_samples_per_class, radius=1.0):
    np.random.seed(0)
    
    # Generate data for cats
    # cats = np.random.rand(num_samples_per_class, 2) * 2 - 1
    cats = make_blobs(n_samples=num_samples_per_class, centers=1, random_state=0, cluster_std=0.40)
    cat_labels = np.zeros(num_samples_per_class)
    
    # Generate data for dogs
    dogs = np.random.rand(num_samples_per_class, 2) * 2.5 + 1.2
    dog_labels = np.ones(num_samples_per_class)
    
    # Generate data for birds
    birds = np.random.rand(num_samples_per_class, 2) * 1.5 + 3.5
    bird_labels = 2 * np.ones(num_samples_per_class)
    
    # Combine the data and labels
    data = np.vstack((cats, dogs, birds))
    labels = np.hstack((cat_labels, dog_labels, bird_labels))
    
    return data, labels


# Train SVM for binary classification
def train_svm(data, labels, kernel='linear'):
    svm_model = svm.SVC(kernel=kernel)
    svm_model.fit(data, labels)
    return svm_model

# Plot data and decision boundary
def plot_data_and_decision_boundary(data, labels, svm_model, new_point=None , text=None):
    plt.scatter(data[labels == 0][:, 0], data[labels == 0][:, 1], color='purple', label='Cats')
    plt.scatter(data[labels == 1][:, 0], data[labels == 1][:, 1], color='red', label='Dogs')
    plt.scatter(data[labels == 2][:, 0], data[labels == 2][:, 1], color='green', label='Birds')


    if text is not None:
        plot_Text = "New Point = " + text

    if new_point is not None:
        plt.scatter(new_point[0, 0], new_point[0, 1], color='black', label=plot_Text)

    # ax = plt.gca()
    # # Generate a mesh grid
    # xx, yy = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 100),
    #                     np.linspace(data[:, 1].min(), data[:, 1].max(), 100))

    # # Stack the mesh grid to create input for decision function
    # xy = np.vstack([xx.ravel(), yy.ravel()]).T

    # # Use the trained SVM model to predict the decision function values
    # Z = svm_model.decision_function(xy)
    # Z = Z.reshape(xx.shape)  # Reshape to match the dimensions of xx and yy

    # # Plot the contour lines
    # ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM: Cats vs Dogs')
    plt.legend()
    plt.show()

# Classify new point
def classify_new_point(svm_model, data, labels):
    x = random.random()*2.5
    y = random.random()*2.5
    new_point = np.array([[x, y]])
    prediction = svm_model.predict(new_point)
    print('The SVM classifies the point as:', 'Dog' if prediction == 1 else 'Cat')
    text = 'Dog' if prediction == 1 else 'Cat'
    plot_data_and_decision_boundary(data, labels, svm_model, new_point, text)

# Main function
def main():
    # Generate synthetic data and train SVM
    num_samples_per_class = 250
    data, labels = generate_synthetic_data(num_samples_per_class)
    svm_model = train_svm(data, labels)

    # Plot the data and decision boundary
    plot_data_and_decision_boundary(data, labels, svm_model)

    while True:
        #Classify new points
        classify_new_point(svm_model, data, labels)

if __name__ == "__main__":
    main()
