import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Generate the data
np.random.seed(42)  # for reproducibility

# Generate 500 points for class 1
mean1 = [1, 0]
cov1 = [[0.3, 0], [0, 0.3]]  # covariance matrix with variance 0.3
class1_points = np.random.multivariate_normal(mean1, cov1, 500)
class1_labels = np.ones(500)

# Generate 500 points for class 2
mean2 = [0, 1]
cov2 = [[0.3, 0], [0, 0.3]]  # covariance matrix with variance 0.3
class2_points = np.random.multivariate_normal(mean2, cov2, 500)
class2_labels = np.zeros(500)

# Combine the data
X = np.vstack((class1_points, class2_points))
y = np.hstack((class1_labels, class2_labels))

# Step 2: Split the data into training and testing sets
X_train_class1, X_test_class1, y_train_class1, y_test_class1 = train_test_split(class1_points, class1_labels, train_size=300, random_state=42)
X_train_class2, X_test_class2, y_train_class2, y_test_class2 = train_test_split(class2_points, class2_labels, train_size=300, random_state=42)

X_train = np.vstack((X_train_class1, X_train_class2))
y_train = np.hstack((y_train_class1, y_train_class2))
X_test = np.vstack((X_test_class1, X_test_class2))
y_test = np.hstack((y_test_class1, y_test_class2))

# Step 3: Train a classifier
# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM classifier
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Step 4: Test the classifier
y_pred = classifier.predict(X_test)

# Print the classification report and accuracy to a text file
with open("classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nAccuracy: " + str(accuracy_score(y_test, y_pred)))

# Optional: Plot the data and decision boundary
def plot_decision_boundary(clf, X, y):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

# Save the plot
plot_decision_boundary(classifier, X_test, y_test)
plt.savefig("decision_boundary.png")
plt.close()
