"""
Created by Mazharul Islam Leon
"""
# Import libraries
import numpy as np
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# set the title of our application
st.title("Machine Learning Web App")
# Writing Markdown text
st.write("# EDA of Traditional ML Algorithm")
# For normal text
st.write("Which ml algorithm showing best result?")

# if we want to use sidebar then just write st.sidebar

# creating a choice/select options
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
st.write("Selected dataset name is :", dataset_name)

# For choosing the ML classifier
classifier_name = st.sidebar.selectbox("Select Dataset", ("KNN", "SVM", "Random Forest"))
st.write("Selected classifier name is :", classifier_name)


# Now load the dataset from scikit-learn according to dataset name
def load_dataset(dataset_name):
    if dataset_name == "KNN":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


# call load dataset function and pass dataset name as parameter
X, y = load_dataset(dataset_name)
st.write("Dataset Shape: ", X.shape)
st.write("Number of Classes: ", len(np.unique(y)))


# Get different parameter according to Classifier
def load_classifier_param(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("no. of estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params


# Call the load_classifier_param function and pass clf_name
params = load_classifier_param(classifier_name)
# st.write("Parameter value is: ", params)


# Now we need to get the classifier algorithm, Additionally pass the params
def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"],
                                     n_estimators=params["n_estimators"],
                                     random_state=123)
    return clf


# Call get_classifier function
clf = get_classifier(classifier_name, params)

# Split our dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123)

# Train our classifier
clf.fit(X_train, y_train)
# Predict method
y_pred = clf.predict(X_test)

# accuracy calculation
accuracy = accuracy_score(y_test, y_pred)
st.write("Classifier:{} and  accuracy: {:.3f}".format(
    classifier_name, accuracy
))

# Plotting the dataset.
# Feature reduction method that transform our dataset into lower dimension
# Unsupervised technique don't need the label
pca = PCA(2)
X_projected = pca.fit_transform(X)
x_1 = X_projected[:, 0]
x_2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(x_1, x_2, c=y, alpha=0.5, cmap="viridis")
plt.title("Dataset : {}".format(dataset_name))
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar()
# Show the plt
st.pyplot(fig)

#