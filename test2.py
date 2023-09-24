import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Generate synthetic data
def generate_synthetic_data(num_samples_per_class):
    np.random.seed(0)

    # Generate cats in a circle
    cat_theta = np.linspace(0, 2 * np.pi, num_samples_per_class)
    cat_r = np.random.uniform(0, 0.5, num_samples_per_class)  # Radius within a circle
    cats_x = cat_r * np.cos(cat_theta)
    cats_y = cat_r * np.sin(cat_theta)
    cats = np.column_stack((cats_x, cats_y))
    cat_labels = np.zeros(num_samples_per_class)

    # Generate dogs around the circumference
    dog_theta = np.linspace(0, 2 * np.pi, num_samples_per_class)
    dog_r = np.random.uniform(0.7, 1.2, num_samples_per_class)  # Radius around the circle
    dogs_x = dog_r * np.cos(dog_theta)
    dogs_y = dog_r * np.sin(dog_theta)
    dogs = np.column_stack((dogs_x, dogs_y))
    dog_labels = np.ones(num_samples_per_class)

    data = np.vstack((cats, dogs))
    labels = np.hstack((cat_labels, dog_labels))
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

    if text is not None:
        plot_Text = "New Point = " + text

    if new_point is not None:
        plt.scatter(new_point[0, 0], new_point[0, 1], color='black', label=plot_Text)


    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm_model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM: Cats vs Dogs')
    plt.legend()
    plt.show()

# Classify new point
def classify_new_point(svm_model, data, labels):
    x = random.randrange(-1,1)
    y = random.randrange(-1,1)
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
    svm_model = train_svm(data, labels, "rbf")

    # Plot the data and decision boundary
    plot_data_and_decision_boundary(data, labels, svm_model)

    while True:
        #Classify new points
        classify_new_point(svm_model, data, labels)

if __name__ == "__main__":
    main()
