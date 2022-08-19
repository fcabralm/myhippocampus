import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import shapiro
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
import pickle

#Load data:
metabo_adni = pd.read_csv('Final2.csv')

#Preprocessing dataset:
metabo_adni = pd.read_csv('Final2.csv')
metabo_adni = metabo_adni.replace(['NDEF'],0)
metabo_adni = metabo_adni.replace(['TAG'],0)
metabo_adni = metabo_adni.fillna(0)

#Define predictors and class:
X_adni = metabo_adni.iloc[:, 0:228].values
y_adni = metabo_adni.iloc[:, 228].values

#Scaling data:
scaler_adni = StandardScaler()
X_adni = scaler_adni.fit_transform(X_adni)

#Generating models and tuning parameters with GridSearch:

# 1- Desicion Tree
parametros_arvore = {'criterion': ['gini', 'entropy','log_loss],
              'splitter': ['best', 'random'],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}

grid_search_arvore = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros_arvore)
grid_search_arvore.fit(X_adni, y_adni)
melhores_parametros_arvore = grid_search_arvore.best_params_
melhor_resultado_arvore = grid_search_arvore.best_score_
print(melhores_parametros_arvore)
print(melhor_resultado_arvore)
#{'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 10, 'splitter': 'random'}
# 0.5729636560622476

# 2- Random Forest
parametros_random = {'criterion': ['gini', 'entropy'],
              'n_estimators': [10, 40, 100, 150],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10]}
grid_search_random = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros_random)
grid_search_random.fit(X_adni, y_adni)
melhores_parametros_random = grid_search_random.best_params_
melhor_resultado_random = grid_search_random.best_score_
print(melhores_parametros_random)
#{'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 40}
print(melhor_resultado_random)
#0.5869595193538856

# 3- KNN
parametros_knn = {'n_neighbors': [3, 5, 10, 20],
              'p': [1, 2]}
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros_knn)
grid_search_knn.fit(X_adni, y_adni)
melhores_parametros_knn = grid_search_knn.best_params_
melhor_resultado_knn = grid_search_knn.best_score_
print(melhores_parametros_knn) 
#{'n_neighbors': 5, 'p': 2}
print(melhor_resultado_knn)
#0.5294297252043731

#4- SVM 
parametros_svm = {'tol': [0.001, 0.0001, 0.00001],
              'C': [1.0, 1.5, 2.0],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

grid_search_svm = GridSearchCV(estimator=SVC(), param_grid=parametros_svm)
grid_search_svm.fit(X_adni, y_adni)
melhores_parametros_svm = grid_search_svm.best_params_
melhor_resultado_svm = grid_search_svm.best_score_
print(melhores_parametros_svm)
print(melhor_resultado_svm)
#{'C': 1.0, 'kernel': 'rbf', 'tol': 0.001}
#0.5758692012213139

# 5- MLP
parametros_redes = {'activation': ['relu', 'logistic', 'tahn','identity'],
              'solver': ['lbfgs','adam', 'sgd'],
              'batch_size': [10],
              'learning_rate': ['constant','invscaling','adaptive'],
              }
grid_search_redes = GridSearchCV(estimator=MLPClassifier(), param_grid=parametros_redes)
grid_search_redes.fit(X_adni, y_adni)
melhores_parametros_redes = grid_search_redes.best_params_
melhor_resultado_redes = grid_search_redes.best_score_
print(melhores_parametros_redes)
print(melhor_resultado_redes)
#{'activation': 'logistic', 'batch_size': 10, 'learning_rate': 'adaptive', 'solver': 'sgd'}
#0.5715749039692701

# Cross-validation using GridSearch:
resultados_arvore = []
resultados_random_forest = []
resultados_knn = []
resultados_svm = []
resultados_rede_neural = []

for i in range(30):
  print(i)
  kfold = KFold(n_splits=10, shuffle=True, random_state=i)
  arvore = DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, min_samples_split=10, splitter='random')
  scores = cross_val_score(arvore, X_adni, y_adni, cv = kfold)
  resultados_arvore.append(scores.mean())
  random_forest = RandomForestClassifier(criterion = 'entropy', min_samples_leaf = 1, min_samples_split=10, n_estimators = 40)
  scores = cross_val_score(random_forest, X_adni, y_adni, cv = kfold)
  resultados_random_forest.append(scores.mean())
  knn = KNeighborsClassifier()
  scores = cross_val_score(knn, X_adni, y_adni, cv = kfold)
  resultados_knn.append(scores.mean())
  svm = SVC(kernel = 'rbf', C = 1.0, tol = 0.001)
  scores = cross_val_score(svm, X_adni, y_adni, cv = kfold)
  resultados_svm.append(scores.mean())
  rede_neural = MLPClassifier(activation = 'logistic', batch_size = 10, solver = 'sgd')
  scores = cross_val_score(rede_neural, X_adni, y_adni, cv = kfold)
  resultados_rede_neural.append(scores.mean())

resultados = pd.DataFrame({'Arvore': resultados_arvore, 'Random forest': resultados_random_forest,
                           'KNN': resultados_knn,
                           'SVM': resultados_svm, 'Rede neural': resultados_rede_neural})
resultados
print(resultados.describe()) 
resultados.var()
(resultados.std() / resultados.mean()) * 100 

## Testing Normality distribution in classifiers accuracy results: 
shapiro(resultados_arvore), shapiro(resultados_random_forest), shapiro(resultados_knn), shapiro(resultados_svm), shapiro(resultados_rede_neural)
#(ShapiroResult(statistic=0.9837696552276611, pvalue=0.9144353866577148),
 #ShapiroResult(statistic=0.9736077189445496, pvalue=0.6418033838272095),
 #ShapiroResult(statistic=0.9643335342407227, pvalue=0.39766865968704224),
 #ShapiroResult(statistic=0.9605059623718262, pvalue=0.3191172480583191),
 #ShapiroResult(statistic=0.9734175205230713, pvalue=0.6362437605857849))

sns.displot(resultados_arvore, kind = 'kde');
sns.displot(resultados_random_forest, kind = 'kde');
sns.displot(resultados_knn, kind = 'kde');
sns.displot(resultados_svm, kind = 'kde');
sns.displot(resultados_rede_neural, kind = 'kde');

# Null-Hypothesis test using ANOVA followed by Tukey's post test comparing classififer's accuracy results:
_, p = f_oneway(resultados_arvore, resultados_random_forest, resultados_knn, resultados_svm, resultados_rede_neural)
p
alpha = 0.05
if p <= alpha:
  print('Null hypothesis rejected. Results are significantly different')
else:
  print('Null hypothesis rejected. Results are not significantly different.')


resultados_algoritmos = {'accuracy': np.concatenate([resultados_arvore, resultados_random_forest, resultados_knn, resultados_svm, resultados_rede_neural]),
                         'algoritmo': ['arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore', 
                          'random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest', 
                          'knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn', 

                          'svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm',
                          'rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural']}

resultados_algoritmos_df = pd.DataFrame(resultados_algoritmos)
resultados_algoritmos_df

compara_algoritmos = MultiComparison(resultados_algoritmos_df['accuracy'], resultados_algoritmos_df['algoritmo'])
teste_estatistico = compara_algoritmos.tukeyhsd()
print(teste_estatistico)

## Tukey post test:
resultados.mean()
teste_estatistico.plot_simultaneous();

## Saving trained classifiers using Pickle:

classificador_arvore_adni = DecisionTreeClassifier(criterion='gini', min_samples_leaf=10, min_samples_split=10, splitter='random')
classificador_arvore_adni.fit(X_adni, y_adni)

classificador_random_forest_adni = RandomForestClassifier(criterion = 'gini', min_samples_leaf = 5, min_samples_split=10, n_estimators = 150)
classificador_random_forest_adni.fit(X_adni, y_adni)

classificador_knn_adni = KNeighborsClassifier()
classificador_knn_adni.fit(X_adni, y_adni)

classificador_svm_adni = SVC(kernel = 'rbf', C = 1.0, tol = 0.001,probability=True)
classificador_svm_adni.fit(X_adni, y_adni) 

classificador_rede_neural_adni = MLPClassifier(activation = 'logistic', batch_size = 10, solver = 'sgd',learning_rate = 'adaptive')
classificador_rede_neural_adni.fit(X_adni, y_adni)

pickle.dump(classificador_rede_neural_adni, open('rede_neural_finalizado.sav', 'wb'))
pickle.dump(classificador_arvore_adni, open('arvore_finalizado.sav', 'wb'))
pickle.dump(classificador_svm_adni, open('svm_finalizado.sav', 'wb'))
pickle.dump(classificador_random_forest_adni, open('random_forest_finalizado.sav', 'wb'))

## Load trained classifiers:
rede_neural = pickle.load(open('rede_neural_finalizado.sav', 'rb'))
arvore = pickle.load(open('arvore_finalizado.sav', 'rb'))
svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random = pickle.load(open('random_forest_adni.sav', 'rb')) 
random = pickle.load(open('random_forest_finalizado.sav', 'rb'))

# Testing classifiers in an single example from same Dataset:                                
exemplo = X_adni[700] 
exemplo = exemplo.reshape(1,-1) 
rede_neural.predict(exemplo) #AD
                                   
exexmplo2 = X_adni[88]
exexmplo2 =  exexmplo2.reshape(1,-1) 
rede_neural.predict(exexmplo2) #CN

# Combining classifiers:
resposta_rede_neural = rede_neural.predict(exemplo)
resposta_arvore = arvore.predict(exemplo)
resposta_svm = svm.predict(exemplo)
resposta_random = random.predict(exemplo)

resposta_rede_neural[0], resposta_arvore[0], resposta_svm[0], resposta_random[0]

CN = 0
AD = 0

if resposta_rede_neural[0] == 'AD':
  AD += 1
else:
  CN += 1

if resposta_arvore[0] == 'AD':
  AD += 1
else:
  CN += 1

if resposta_svm[0] == 'AD':
  AD += 1
else:
  CN += 1

if resposta_random[0] == 'AD':
  AD += 1
else:
  CN += 1

if CN > AD:
  print('Patient without Alzheimers Disease, as computed by 4 algorithms.')
elif CN == AD:
  print('Cannot determine diagnosis, as computed by 4 algorithms.')
else:
  print('Patient with Alzheimers Disease, as computed by 4 algorithms. ')







