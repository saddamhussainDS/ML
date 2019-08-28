%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Seaborn import for plotting
import seaborn as sns
sns.set()

from sklearn.datasets.samples_generator import make_blobs

X1,y1=make_blobs(n_features=50,centers=2,random_state=0,cluster_std=0.60)
X1.shape,y1.shape

plt.scatter(X1[:,0],X1[:,1],c=y1,s=50,cmap='autumn')
plt.show()

xfit=np.linspace(-1,3.5)
xfit

plt.scatter(X1[:,0],X1[:,1],c=y1,s=50,cmap='autumn')
plt.plot([0.6],[2.1],'x',color='red',mew=2.,markersize=10)

for m,b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit,m*xfit+b,'-k')
    
plt.xlim(-1,3.5)

### Maximizing the margins

xfit=np.linspace(-1,3.5)
plt.scatter(X1[:,0],X1[:,1],c=y1,s=50,cmap='autumn')

for m,b,d in [(1,0.65,0.33),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:
    yfit=m*xfit+b
    plt.plot(xfit,yfit,'-k')
    plt.fill_between(xfit,yfit-d,yfit+d,edgecolor='none',color='RgB',alpha=0.4)
    
plt.xlim(-1,3.5)

### Fitting a support Vector

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

%matplotlib inline
# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

clf.support_vectors_

## Face recognition model 

from sklearn import svm
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline

from sklearn.datasets import fetch_lfw_people

faces=fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


fig,ax=plt.subplots(3,6,figsize=(10,5))
plt.subplots_adjust(wspace=1)
for i,axi in enumerate(ax.flat):
    axi.imshow(faces.images[i],cmap='bone')
    axi.set(xticks=[],yticks=[],xlabel=faces.target_names[faces.target[i]])


from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


pca=PCA(n_components=150,svd_solver='randomized',whiten=True,random_state=42)
svc=SVC(kernel='rbf')
model=make_pipeline(pca,svc)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(faces.data,faces.target, random_state=42)

from sklearn.grid_search import GridSearchCV

param_grid={'svc__C':[1,5,10,15,20],'svc__gamma':[0.001,0.005,0.0001,0.0005]}
grid=GridSearchCV(model,param_grid)

%time grid.fit(X_train,y_train)

## Cross validation using GridSearchCV

grid.best_params_

model=grid.best_estimator_
model

yfit=model.predict(X_test)

model.score(X_test,y_test)

fig,ax=plt.subplots(4,6)
for i ,axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62,47),cmap='bone')
    axi.set(xticks=[],yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],color='black' if yfit[i]==y_test[i] else 'red')

from sklearn.metrics import classification_report
print(classification_report(y_test,yfit,target_names=faces.target_names))

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test,yfit)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,xticklabels=faces.target_names,yticklabels=faces.target_names)
plt.xlabel('True label')
plt.ylabel('Predicted label')
