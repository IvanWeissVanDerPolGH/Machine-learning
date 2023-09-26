import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from pointList import pointList
from pointObject import PointObject
import pandas as pd

def generate_scattered_points(num_points, center_x=0, center_y=0, min_dist=0, max_dist=20):
    """
    Generates a list of scattered X, Y points around the center.

    Parameters:
    num_points (int): Number of points to generate.
    center_x (float): X coordinate of the center. Default is 0.
    center_y (float): Y coordinate of the center. Default is 0.
    max_dist (float): Maximum scatter distance from the center. Default is 20.

    Returns:
    list: List of tuples representing X, Y points.
    """
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    distances = np.random.uniform(min_dist, max_dist, num_points)

    x_points = center_x + distances * np.cos(angles)
    y_points = center_y + distances * np.sin(angles)

    scattered_points = list(zip(x_points, y_points))
    return scattered_points

def train_svm_model_for_groups(df_points,kernel="linear"):
    """
    Train SVM models for each group and return a dictionary with trained models.

    Parameters:
    df_points (DataFrame): DataFrame with columns 'Data_x', 'Data_y', 'name', and 'color'.

    Returns:
    dict: A dictionary with group names as keys and trained SVM models as values.
    """
    X_complete = df_points[['Data_x', 'Data_y']].values
    y_complete = LabelEncoder().fit_transform(df_points['name'])

    clf = svm.SVC(kernel=kernel)
    clf.fit(X_complete, y_complete)

    unique_groups = df_points['name'].unique()
    svm_models = {group: {'model': clf, 'color': df_points[df_points['name'] == group]['color'].iloc[0]} for group in unique_groups}
    return svm_models



def plot_scattered_points(df_points):
    """
    Plots the scattered points using Matplotlib.

    Parameters:
    df_points (DataFrame): DataFrame with columns 'Data_x', 'Data_y', 'name', and 'color'.
    """
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scattered Points')
    unique_groups = df_points['name'].unique()
    for group in unique_groups:
        group_data = df_points[df_points['name'] == group]
        x_points, y_points = group_data['Data_x'], group_data['Data_y']
        color = group_data['color'].iloc[0]  # Use the first color for the group
        plt.scatter(x_points, y_points, color=color, label=f'Scattered Points {group}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_groups_with_decision_boundary(df_points, trained_models, new_point = None, text= None):
    """
    Plot the groups with SVM decision boundaries.

    Parameters:
    df_points (DataFrame): DataFrame with columns 'Data_x', 'Data_y', 'name', and 'color'.
    trained_models (dict): A dictionary with group names as keys and trained SVM models as values.
    """
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    
   
    
    x_min, x_max = df_points['Data_x'].min() - 1, df_points['Data_x'].max() + 1
    y_min, y_max = df_points['Data_y'].min() - 1, df_points['Data_y'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    for group in df_points['name'].unique():
        X = df_points[df_points['name'] == group][['Data_x', 'Data_y']].values
        y = LabelEncoder().fit_transform([group] * len(X))
        model_info = trained_models[group]
        clf = model_info['model']
        color = model_info['color']

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap([color]), edgecolors='k', label=f'Group {group}')

        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors=[color], levels=[0], linestyles=['-'], linewidths=[2])


    if text is not None:
        plot_Text = "New Point = " + text

    if new_point is not None:
        plt.scatter(new_point[0], new_point[1], color='black', label=plot_Text)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('SVM Decision Boundaries')
    plt.legend()
    plt.show()

def create_dataframe(points_data, group_info):
    """
    Creates a DataFrame from points data and group information.

    Parameters:
    points_data (list): List of tuples containing points data (x, y).
    group_info (dict): Dictionary containing group name and color.

    Returns:
    pd.DataFrame: DataFrame containing the points data and group information.
    """
    df = pd.DataFrame(points_data, columns=['Data_x', 'Data_y'])
    df['name'] = group_info['name']
    df['color'] = group_info['color']
    return df

def print_dataframe(df_points):
    print(df_points.to_markdown())

def classify_new_point(df_points, trained_models, svm_model, labels):
    # Generate a new point
    new_point = generate_scattered_points(1, center_x=0, center_y=0, min_dist=0, max_dist=20)[0]  # Adjusted parameters

    # Predict the label for the new point
    prediction = svm_model.predict([new_point])

    # Determine the classification result message based on the prediction
    if prediction == 0:
        result_message = f'The SVM classifies the point as: {labels[0]["name"]}'
        text = f' {labels[0]["name"]}'
    else:
        result_message = f'The SVM classifies the point as: {labels[1]["name"]}'
        text = f' {labels[1]["name"]}'

    # print(result_message)
    # print(text)
    plot_groups_with_decision_boundary(df_points, trained_models, new_point = new_point, text= text)