from sklearn import datasets
from sklearn.decomposition import PCA


iris = datasets.load_iris()

x = iris.data
print x
y = iris.target
print y
target_names = iris.target_names
print target_names

pca = PCA(n_components=3)
x_r = pca.fit(x).transform(x)
print x_r
