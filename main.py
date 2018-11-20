import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering, KMeans
import keras
from keras.models import Sequential
from keras.layers import Dense

# Read the data from the file
data = pd.read_csv('abalone.data', header=None)
header = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
          'Rings']
data.columns = header

# Splitting data into attributes and target value
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

# Converting Sex attribute into classes
label_encoder = LabelEncoder()
X[:, 0] = label_encoder.fit_transform(X[:, 0])
# Converting Sex classes into dummy variables
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()

'''Outliers Detection'''
clf = IsolationForest(max_samples=100, random_state=np.random.RandomState(42))
clf.fit(X)
outlier = clf.predict(X)
X_test = X[outlier == 1]
Y_test = Y[outlier == 1]

'''Data standardization'''
sc = StandardScaler()
X[:, 2:] = sc.fit_transform(X[:, 2:])

'''Dimension reduction algorithms provide us find strong correlations between attributes and use for our models
    only that attributes which changes will affect our target value
    The ones that we won't use have a slight influence so we can remove them from our dataset'''

'''PCA dimension reduction: finding the strongest linear correlation between attribute and target value'''
pca = PCA(n_components=2)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_  # is used to select right number of components

'''Kernel PCA dimension reduction: finding the strongest non-linear correlation between attribute and target value'''
# kpca = KernelPCA(n_components=3, kernel='rbf')
# X = kpca.fit_transform(X)
# explained_variance = np.var(X, axis=0)
# is used to select right number of components
# explained_variance_ratio = explained_variance / np.sum(explained_variance)

# Splitting dataset into training and test sets with 80/20 relation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

'''Random Tree Classification
    Model is based on creating more than one decision tree (n_estimators)
    Then it creates each decision tree from the random chosen data (k subset of our data)
    When new attributes come each decision tree determine the class it belongs to. Majority of votes wins'''
rfr = RandomForestClassifier(n_estimators=64)
rfr.fit(X_train, Y_train)
rfr_predictions = rfr.predict(X_test)
rfr_predictions = np.reshape(rfr_predictions, (-1, 1))

accuracy_score(Y_test, rfr_predictions)
accuracies_rfc = cross_val_score(estimator=rfr, X=X_train, y=Y_train, cv=10)

'''Artificial Neural Network Classification
    Neural Networks are made of neurons that have inputs and activation functions and of the layers of this neurons
    that are connected between each other. Typically ann is made of input, output and hidden layers. Output layers
    are learned from the output they should give (our training set) - they compare the output they get and the output
    they should have and compare them. The error is delivered to the connected neurons so the weights they have
    are changing - called backpropagation'''

# Converting target value into dummy variables for ANN classification
one_hot_encoder = OneHotEncoder(categorical_features=[0])
Y_nn = one_hot_encoder.fit_transform(np.reshape(Y, (-1, 1))).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_nn, test_size=0.2, random_state=0)

# Initialising the ANN
classifier = Sequential()

# The input layer changes depending which dimensionally reduction function we are using
# 3 in case KernelPCA and 2 in case PCA.
# classifier.add(Dense(units=18, kernel_initializer='uniform', activation='relu', input_dim=3))
classifier.add(Dense(units=18, kernel_initializer='uniform', activation='relu', input_dim=2))
classifier.add(Dense(units=11, kernel_initializer='uniform', activation='softmax'))
classifier.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=28, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, epochs=150)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
predictions = np.array([[1.0 if j == i.max() else 0.0 for j in i] for i in y_pred])
accuracy_score(Y_test, predictions)
'''Clustering 
    Clustering is used to split similar data into the groups which can be used for class reduction that can
    increase the accuracy of classification models
    Clustering is the type of unsupervised learning - no need of test set'''
'''KMeans Clustering
    This algorithm choose a n_clusters centroid (they are chosen by k means ++ algorithm which preserve us
    from the random initialization trap). Then we assign all the points we have to this centroids. After 
    we change their coordinates on the average sum of the assigned coordinates. Repeating this steps until
    the points stop reassigning
    To find the optimal number of clusters we are using within cluster sum of squares (The sum of the squared deviations
     from each observation and the cluster centroid.)'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plot.plot(range(1, 11), wcss)
plot.show()

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = kmeans.fit_predict(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y_means, test_size=0.2, random_state=0)

'''Using some other classification models to prove that accuracy of prediction clustered target value is improved'''

'''Random Forest Classification'''
rfc = RandomForestClassifier(n_estimators=64)
rfc.fit(X_train, Y_train)
rfc_pred = rfc.predict(X_test)

accuracy_score(Y_test, rfc_pred)

'''KNN Classification '''
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, Y_train)
knn_prediction = knn_classifier.predict(X_test)

accuracy_score(Y_test, knn_prediction)

'''SVC Classification'''
svc_classifier = SVC(C=2.0)
svc_classifier.fit(X_train, Y_train)
svc_prediction = svc_classifier.predict(X_test)

accuracy_score(Y_test, svc_prediction)

'''k-fold cross validation is based on splitting a dataset into cv numbers of folds, 1 of them is using for testing
    9 of them is for training. And we are continuing until each of the folds is used for testing our model'''
'''k-fold cross validation'''
accuracies_rfc = cross_val_score(estimator=rfc, X=X, y=y_means, cv=10)
accuracies_knn = cross_val_score(estimator=knn_classifier, X=X, y=y_means, cv=10)
accuracies_svc = cross_val_score(estimator=svc_classifier, X=X, y=y_means, cv=10)


'''Hierarchical Clustering'''
'''This algorithm based on making dendogram. First we are considering each point as a separate cluster. Then we  
    union the closest clusters into one and continuing until there is one cluster left. 
    To find the optimal number of clusters we should look at dendogram we made and choose the longest vertical line 
    that isn't crossing any horizontal. Than draw a horizontal perpendicular line at the middle - number of crossed 
    lines is a number of clusters we should have'''
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plot.title('Dendogram')
plot.show()

hierc = AgglomerativeClustering(n_clusters=2)
y_cl = hierc.fit_predict(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y_cl, test_size=0.2, random_state=0)

'''Random Forest Classification'''
rfc = RandomForestClassifier(n_estimators=64)
rfc.fit(X_train, Y_train)
rfc_pred = rfc.predict(X_test)

accuracy_score(Y_test, rfc_pred)

'''KNN Classification '''
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, Y_train)
knn_prediction = knn_classifier.predict(X_test)

accuracy_score(Y_test, knn_prediction)

'''SVC Classification'''
svc_classifier = SVC(C=2.0)
svc_classifier.fit(X_train, Y_train)
svc_prediction = svc_classifier.predict(X_test)

accuracy_score(Y_test, svc_prediction)

'''k-fold cross validation'''
accuracies_rfc_hierarchical = cross_val_score(estimator=rfc, X=X, y=y_cl, cv=10)
accuracies_knn_hierarchical = cross_val_score(estimator=knn_classifier, X=X, y=y_cl, cv=10)
accuracies_svc_hierarchical = cross_val_score(estimator=svc_classifier, X=X, y=y_cl, cv=10)
