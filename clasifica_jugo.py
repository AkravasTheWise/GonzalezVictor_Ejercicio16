import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
# Carga datos
data = pd.read_csv('OJ.csv')

# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

data['Target'] = purchasebin

# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)

# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')
print(predictors)

data_train, data_test, target_train, target_test = train_test_split(data, data['Target'], train_size=0.5)
import sklearn.tree
depth=10
iteraciones=100
f1Metrics_train=np.zeros((depth,iteraciones))
f1Metrics_test=np.zeros((depth,iteraciones))
averageImportance=np.zeros((depth))
for i in range(depth):
    for j in range(iteraciones):
        d_train=resample(data_train)
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=i+1)
        clf.fit(d_train, d_train['Target'])
        f1Metrics_train[i][j]=sklearn.metrics.f1_score(data_train['Target'], clf.predict(d_train))
        
        
        d_test=resample(data_test)
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=i+1)
        clf.fit(d_test, d_test['Target'])
        f1Metrics_test[i][j]=sklearn.metrics.f1_score(data_test['Target'], clf.predict(d_test))

plt.errorbar(x=[1,2,3,4,5,6,7,8,9,10],y=np.mean(f1Metrics_train,axis=1),yerr=np.std(f1Metrics_train,axis=1))
plt.errorbar(x=[1,2,3,4,5,6,7,8,9,10],y=np.mean(f1Metrics_test,axis=1),yerr=np.std(f1Metrics_test,axis=1))
plt.xlabel('max depth')
plt.ylabel('Average F1-score')
plt.savefig('F1_training_test.png')
