from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21, stratify=y)
print(type(X_train))
print(type(X_test))
model = KMeans(n_clusters=3)
model.fit(X_train)
labels = model.predict(X_test)
print(labels)
centeroids = model.cluster_centers_
centroids_sl = centeroids[:, 0]
centroids_sw = centeroids[:, 1]
centroids_pl = centeroids[:, 2]
centroids_pw = centeroids[:, 3]
print(centroids_sl)
print(centroids_sw)
print(centroids_pl)
print(centroids_pw)
print(centeroids)
