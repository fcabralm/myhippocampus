
## Based on: https://analyticsindiamag.com/a-guide-to-explaining-feature-importance-in-neural-networks-using-shap/

#!pip install shap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
plt.style.use('fivethirtyeight')
import sklearn
import lifelines
import shap
import time
shap.initjs()

# Open and pre-process data:
metabo_adni = pd.read_csv('Final2.csv')
metabo_adni = metabo_adni.replace(['NDEF'],0)
metabo_adni = metabo_adni.replace(['TAG'],0)
metabo_adni = metabo_adni.fillna(0)

# Define predictors and class:
X= metabo_adni.iloc[:, 0:228]
y= metabo_adni.iloc[:, 228]

# Creating array with features name
features = list(metabo_adni.columns)  
features = features[0:228]

#Scale the data:
scaler_adni = StandardScaler()
X = scaler_adni.fit_transform(X)

#Load previously generated models: 
rede_neural = pickle.load(open('rede_neural_finalizado.sav', 'rb'))
svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random = pickle.load(open('random_forest_finalizado.sav', 'rb')) 

## Testing MLP classifier:
example = X[588] # a linha saiu em forma de coluna !
example = example.reshape(1,-1) # para fazer o trasnpose!
rede_neural.predict(example)

# This will indicate in shap values who is 0 and who is 1:
rede_neural.predict_proba(example)
#example, if algoirthms indicates AD and it is correct, check the probabilities
# the greatest proba in this example will be in 0 position, thus 0 = AD
 
# Obtaining SHAP values:
#### Decrease computing time using k meann sampling from predictors (for large datasets like this)
X_summary = shap.kmeans(X, 100)

#RANDOM FOREST  
### Explain all the predictions in the test set
explainer_arvore = shap.KernelExplainer(random.predict_proba, X_summary)
shap_values = explainer_arvore.shap_values(X)

# Summarize all prediction's explanations
shap.summary_plot(shap_values, X, features) # In bars
shap.summary_plot(shap_values[0], X, show=False, max_display=X.shape[0], feature_names= features)
shap.summary_plot(shap_values[0], X, feature_names=features)# for AD # Check explanation above in classifier evaluation
shap.summary_plot(shap_values[1], X, feature_names=features)# for CN

#Dependence plot:
shap.dependence_plot("GP", shap_values[0], X, features)
shap.dependence_plot("GLY", shap_values[0], X, features)

# visualize the first prediction's explanation with a force plot
shap.force_plot(explainer_arvore.expected_value[0], shap_values[0][0], features = features)
#Error

##### SVM ######
#explain all the predictions in the test set
explainer_svm = shap.KernelExplainer(svm.predict_proba, X_summary)
shap_values = explainer_svm.shap_values(X)

# Visualize all features:
shap.summary_plot(shap_values, X, features) # com nome dos features
shap.summary_plot(shap_values[0], X, feature_names=features)# for AD # Check explanation above in classifier evaluation
shap.summary_plot(shap_values[0], X, show=False, max_display=X.shape[0], feature_names= features) # aqui mostra todos os fetures

#Dependence plot:
shap.dependence_plot("GP", shap_values[0], X, features)
shap.dependence_plot("GLOL", shap_values[0], X, features)
shap.dependence_plot("TYR", shap_values[0], X, features)
shap.dependence_plot("CREA", shap_values[0], X, features)

# Visualize the first prediction's explanation with a force plot
shap.force_plot(explainer_svm.expected_value[0], shap_values[0][0], features = features)

#### ######    MLP       ################
# explain all the predictions in the test set
explainer_rede_neural = shap.KernelExplainer(rede_neural.predict_proba, X_summary)
shap_values = explainer_rede_neural.shap_values(X)

#Visualize all features:
shap.summary_plot(shap_values, X, features) # In bars
shap.summary_plot(shap_values[0], X, show=False, max_display=X.shape[0], feature_names= features)
shap.summary_plot(shap_values[0], X, feature_names=features)# for AD # Check explanation above in classifier evaluation
shap.summary_plot(shap_values[1], X, feature_names=features)# for CN

#Dependence plot:
shap.dependence_plot("GP", shap_values[0], X, features)
shap.dependence_plot("GLOL", shap_values[0], X, features)
shap.dependence_plot("LA.FA", shap_values[0], X, features)
shap.dependence_plot("CREA", shap_values[0], X, features)




















