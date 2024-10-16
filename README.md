# K-Means-Clustering-and-Linear-Regression-
This repository contains two machine learning implementations: Linear Regression (a supervised learning algorithm) and K-Means Clustering (an unsupervised learning algorithm). The implementations are designed to predict a person's weight based on their age and height using Linear Regression and to group people based on their height and weight using K-Means Clustering.

Table of Contents

    Introduction
    Requirements
    Linear Regression
        Dataset
        Implementation Steps
        Expected Output
    K-Means Clustering
        Dataset
        Implementation Steps
        Expected Output
    How to Run the Code
    License

Introduction

This project demonstrates basic machine learning techniques using Python. The two main algorithms implemented are:

Linear Regression: A supervised learning technique to predict continuous variables (e.g., predicting weight based on age and height).
K-Means Clustering: An unsupervised learning algorithm used to group data points into clusters (e.g., grouping individuals based on height and weight).

The project uses the following Python libraries:

    numpy
    pandas
    matplotlib
    scikit-learn

Requirements

Before running the code, ensure that the following libraries are installed in your Python environment:

      pip install numpy pandas scikit-learn matplotlib

Alternatively, if you are using a virtual environment, make sure it is activated and that the required libraries are installed in the virtual environment.

Linear Regression

Dataset

The dataset consists of three features: Age, Height, and Weight. We will use the Age and Height as the input features (X) and predict Weight as the target variable (Y).

        Age	 Height (cm)	Weight (kg)
        3	    80	          12
        6	    100	          22
        9	    120	          35
        12	  135	          50
        15	  160	          60
        18	  175	          70

Implementation Steps

Dataset Creation: Manually create a dataset for Age, Height, and Weight.

Train-Test Split: Split the dataset into training and testing sets to evaluate model performance.

Linear Regression Model: Train the model on the training data.

Prediction: Make predictions using the test data.

Evaluation: Use the Mean Squared Error (MSE) metric to evaluate the model's performance.

Visualization: Create a scatter plot to visualize the actual and predicted weights.

Expected Output

Mean Squared Error: A numerical value indicating the average squared difference between the actual and predicted weights.

Scatter Plot: A plot showing the actual vs. predicted weights.

Example Output (Console):

    Mean Squared Error: 1.2345
    Predicted weights: [some array of predicted values]

Example Output (Plot):

A scatter plot showing:

-Blue dots for the actual weights.

-Red dots for the predicted weights based on the test set.

K-Means Clustering
Dataset

The dataset used for K-Means Clustering contains only Height and Weight features, which are used to group individuals into clusters.

      Height (cm)	Weight (kg)
      80	          12
      100	          22
      120	          35
      135	          50
      160	          60
      175	          70
      140	          45
      155	          65
      180	          85
      110	          30

Implementation Steps:

Dataset Creation: Create a dataset for Height and Weight.

K-Means Clustering: Apply the K-Means algorithm to cluster the data into k clusters (where k=3 in this example).

Cluster Visualization: Plot the clustered data with different colors representing different clusters.

Output Cluster Centers: Print the centers of the clusters.

Assign Clusters: Assign each data point to one of the clusters and print the resulting groups.

Expected Output:

Cluster Centers: Coordinates representing the center of each cluster.

Scatter Plot: A plot showing the clustered data points, with each cluster color-coded.

Example Output (Console):

      Cluster Centers:
       [[112.5, 22.5]
        [165.0, 70.0]
        [132.5, 47.5]]
      Assigned Clusters:
        Height  Weight  Cluster
      0      80      12        0
      1     100      22        0
      2     120      35        2
      ...

How to Run the Code:

1. Clone the Repository:
   
      git clone [repository_url]
      cd [repository_name]

3. Create a Virtual Environment:
   
       python -m venv venv
   
5. Activate the Virtual Environment:
   
On Windows: 

      .\venv\Scripts\activate
      
On macOS/Linux: 

      source venv/bin/activate
   
7. Install the Requirements:

        pip install -r requirements.txt
   
9. Run the Applications:

For Linear Regression:  

      python linear_regression_weight_prediction.py
      
For K-Means Clustering:  

      python kmeans_clustering.py

The results for both models will be displayed in the terminal, and plots will be shown to visualize the output.

