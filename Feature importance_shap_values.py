#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 20:24:39 2022

@author: felipecabralmiranda
"""
##https://analyticsindiamag.com/a-guide-to-explaining-feature-importance-in-neural-networks-using-shap/

#!pip install shap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import sklearn
import lifelines
import shap
import time
shap.initjs()
# This sets a common size for all the figures we will draw.
#plt.rcParams['figure.figsize'] = [20, 20]

#convert the data into array
#metabo_adni = metabo_adni.values

#Select X and y values
X= metabo_adni.iloc[:, 0:228]
#X = np.asarray(X).astype('float32') 

y= metabo_adni.iloc[:, 228]

#creating array with features name
metabo_adni.to_records(index=True) # a saida deste codigo sao todos os records do datafram, copiei ecolei os cabecalhos aqui embaixo:
features = [('XXL.VLDL.P', '<f8'), ('XXL.VLDL.L', '<f8'), ('XXL.VLDL.PL', '<f8'), ('XXL.VLDL.C', '<f8'), ('XXL.VLDL.CE', '<f8'), ('XXL.VLDL.FC', '<f8'), ('XXL.VLDL.TG', '<f8'), ('XL.VLDL.P', '<f8'), ('XL.VLDL.L', '<f8'), ('XL.VLDL.PL', '<f8'), ('XL.VLDL.C', '<f8'), ('XL.VLDL.CE', '<f8'), ('XL.VLDL.FC', '<f8'), ('XL.VLDL.TG', '<f8'), ('L.VLDL.P', '<f8'), ('L.VLDL.L', '<f8'), ('L.VLDL.PL', '<f8'), ('L.VLDL.C', '<f8'), ('L.VLDL.CE', '<f8'), ('L.VLDL.FC', '<f8'), ('L.VLDL.TG', '<f8'), ('M.VLDL.P', '<f8'), ('M.VLDL.L', '<f8'), ('M.VLDL.PL', '<f8'), ('M.VLDL.C', '<f8'), ('M.VLDL.CE', '<f8'), ('M.VLDL.FC', '<f8'), ('M.VLDL.TG', '<f8'), ('S.VLDL.P', '<f8'), ('S.VLDL.L', '<f8'), ('S.VLDL.PL', '<f8'), ('S.VLDL.C', '<f8'), ('S.VLDL.CE', '<f8'), ('S.VLDL.FC', '<f8'), ('S.VLDL.TG', '<f8'), ('XS.VLDL.P', '<f8'), ('XS.VLDL.L', '<f8'), ('XS.VLDL.PL', '<f8'), ('XS.VLDL.C', '<f8'), ('XS.VLDL.CE', '<f8'), ('XS.VLDL.FC', '<f8'), ('XS.VLDL.TG', '<f8'), ('IDL.P', '<f8'), ('IDL.L', '<f8'), ('IDL.PL', '<f8'), ('IDL.C', '<f8'), ('IDL.CE', '<f8'), ('IDL.FC', '<f8'), ('IDL.TG', '<f8'), ('L.LDL.P', '<f8'), ('L.LDL.L', '<f8'), ('L.LDL.PL', '<f8'), ('L.LDL.C', '<f8'), ('L.LDL.CE', '<f8'), ('L.LDL.FC', '<f8'), ('L.LDL.TG', '<f8'), ('M.LDL.P', '<f8'), ('M.LDL.L', '<f8'), ('M.LDL.PL', '<f8'), ('M.LDL.C', '<f8'), ('M.LDL.CE', '<f8'), ('M.LDL.FC', '<f8'), ('M.LDL.TG', '<f8'), ('S.LDL.P', '<f8'), ('S.LDL.L', '<f8'), ('S.LDL.PL', '<f8'), ('S.LDL.C', '<f8'), ('S.LDL.CE', '<f8'), ('S.LDL.FC', '<f8'), ('S.LDL.TG', '<f8'), ('XL.HDL.P', '<f8'), ('XL.HDL.L', '<f8'), ('XL.HDL.PL', '<f8'), ('XL.HDL.C', '<f8'), ('XL.HDL.CE', '<f8'), ('XL.HDL.FC', '<f8'), ('XL.HDL.TG', '<f8'), ('L.HDL.P', '<f8'), ('L.HDL.L', '<f8'), ('L.HDL.PL', '<f8'), ('L.HDL.C', '<f8'), ('L.HDL.CE', '<f8'), ('L.HDL.FC', '<f8'), ('L.HDL.TG', '<f8'), ('M.HDL.P', '<f8'), ('M.HDL.L', '<f8'), ('M.HDL.PL', '<f8'), ('M.HDL.C', '<f8'), ('M.HDL.CE', '<f8'), ('M.HDL.FC', '<f8'), ('M.HDL.TG', '<f8'), ('S.HDL.P', '<f8'), ('S.HDL.L', '<f8'), ('S.HDL.PL', '<f8'), ('S.HDL.C', '<f8'), ('S.HDL.CE', '<f8'), ('S.HDL.FC', '<f8'), ('S.HDL.TG', '<f8'), ('XXL.VLDL.PL_.', 'O'), ('XXL.VLDL.C_.', 'O'), ('XXL.VLDL.CE_.', 'O'), ('XXL.VLDL.FC_.', 'O'), ('XXL.VLDL.TG_.', 'O'), ('XL.VLDL.PL_.', 'O'), ('XL.VLDL.C_.', 'O'), ('XL.VLDL.CE_.', 'O'), ('XL.VLDL.FC_.', 'O'), ('XL.VLDL.TG_.', 'O'), ('L.VLDL.PL_.', 'O'), ('L.VLDL.C_.', 'O'), ('L.VLDL.CE_.', 'O'), ('L.VLDL.FC_.', 'O'), ('L.VLDL.TG_.', 'O'), ('M.VLDL.PL_.', '<f8'), ('M.VLDL.C_.', '<f8'), ('M.VLDL.CE_.', '<f8'), ('M.VLDL.FC_.', '<f8'), ('M.VLDL.TG_.', '<f8'), ('S.VLDL.PL_.', '<f8'), ('S.VLDL.C_.', '<f8'), ('S.VLDL.CE_.', '<f8'), ('S.VLDL.FC_.', '<f8'), ('S.VLDL.TG_.', '<f8'), ('XS.VLDL.PL_.', '<f8'), ('XS.VLDL.C_.', '<f8'), ('XS.VLDL.CE_.', '<f8'), ('XS.VLDL.FC_.', '<f8'), ('XS.VLDL.TG_.', '<f8'), ('IDL.PL_.', '<f8'), ('IDL.C_.', '<f8'), ('IDL.CE_.', '<f8'), ('IDL.FC_.', '<f8'), ('IDL.TG_.', '<f8'), ('L.LDL.PL_.', '<f8'), ('L.LDL.C_.', '<f8'), ('L.LDL.CE_.', '<f8'), ('L.LDL.FC_.', '<f8'), ('L.LDL.TG_.', '<f8'), ('M.LDL.PL_.', '<f8'), ('M.LDL.C_.', '<f8'), ('M.LDL.CE_.', '<f8'), ('M.LDL.FC_.', '<f8'), ('M.LDL.TG_.', '<f8'), ('S.LDL.PL_.', '<f8'), ('S.LDL.C_.', '<f8'), ('S.LDL.CE_.', '<f8'), ('S.LDL.FC_.', '<f8'), ('S.LDL.TG_.', '<f8'), ('XL.HDL.PL_.', 'O'), ('XL.HDL.C_.', 'O'), ('XL.HDL.CE_.', 'O'), ('XL.HDL.FC_.', 'O'), ('XL.HDL.TG_.', 'O'), ('L.HDL.PL_.', '<f8'), ('L.HDL.C_.', '<f8'), ('L.HDL.CE_.', '<f8'), ('L.HDL.FC_.', '<f8'), ('L.HDL.TG_.', '<f8'), ('M.HDL.PL_.', '<f8'), ('M.HDL.C_.', '<f8'), ('M.HDL.CE_.', '<f8'), ('M.HDL.FC_.', '<f8'), ('M.HDL.TG_.', '<f8'), ('S.HDL.PL_.', '<f8'), ('S.HDL.C_.', '<f8'), ('S.HDL.CE_.', '<f8'), ('S.HDL.FC_.', '<f8'), ('S.HDL.TG_.', '<f8'), ('VLDL.D', '<f8'), ('LDL.D', '<f8'), ('HDL.D', '<f8'), ('SERUM.C', '<f8'), ('VLDL.C', '<f8'), ('REMNANT.C', '<f8'), ('LDL.C', '<f8'), ('HDL.C', '<f8'), ('HDL2.C', '<f8'), ('HDL3.C', '<f8'), ('ESTC', '<f8'), ('FREEC', '<f8'), ('SERUM.TG', '<f8'), ('VLDL.TG', '<f8'), ('LDL.TG', '<f8'), ('HDL.TG', '<f8'), ('TOTPG', '<f8'), ('TG.PG', '<f8'), ('PC', '<f8'), ('SM', '<f8'), ('TOTCHO', '<f8'), ('APOA1', '<f8'), ('APOB', '<f8'), ('APOB.APOA1', '<f8'), ('TOTFA', '<f8'), ('UNSAT', '<f8'), ('DHA', '<f8'), ('LA', '<f8'), ('FAW3', '<f8'), ('FAW6', '<f8'), ('PUFA', '<f8'), ('MUFA', '<f8'), ('SFA', '<f8'), ('DHA.FA', '<f8'), ('LA.FA', '<f8'), ('FAW3.FA', '<f8'), ('FAW6.FA', '<f8'), ('PUFA.FA', '<f8'), ('MUFA.FA', '<f8'), ('SFA.FA', '<f8'), ('GLC', '<f8'), ('LAC', '<f8'), ('PYR', 'O'), ('CIT', 'O'), ('GLOL', 'O'), ('ALA', '<f8'), ('GLN', 'O'), ('GLY', 'O'), ('HIS', '<f8'), ('ILE', '<f8'), ('LEU', '<f8'), ('VAL', '<f8'), ('PHE', '<f8'), ('TYR', '<f8'), ('ACE', '<f8'), ('ACACE', '<f8'), ('BOHBUT', '<f8'), ('CREA', '<f8'), ('ALB', '<f8'), ('GP', '<f8')]
features = np.array(features)
features = np.delete(features, 1,1) #np.delete (array, linha(0) ou coluna(1), qual delas?)

#process the data
from sklearn.preprocessing import StandardScaler
scaler_adni = StandardScaler()
X = scaler_adni.fit_transform(X)

#split the data 80% training and 20% testing
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 4)

#Load the model 
#carregar classificadores ja treinados no banco de dados (metodo: gridsearch e validacao cruzada)
import pickle
rede_neural = pickle.load(open('rede_neural_finalizado.sav', 'rb'))
#arvore = pickle.load(open('arvore_finalizado.sav', 'rb'))
svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random = pickle.load(open('random_forest_finalizado.sav', 'rb')) 

#https://slundberg.github.io/shap/notebooks/Iris%20classification%20with%20scikit-learn.html
## Testing MLP classifier:
example = X[588] # a linha saiu em forma de coluna !
example = example.reshape(1,-1) # para fazer o trasnpose!

rede_neural.predict(example)
# This will indicate in shap values who is 0 and who is 1:
rede_neural.predict_proba(example)
#example, if algoirthms indicates AD and it is correct, check the probabilities
# the greatest proba in this example will be in 0 position, thus 0 = AD
 
random.predict(example)
svm.predict(example)

##########################################################################################
##########################     RANDOM FOREST     #############################################
##########################################################################################
### DECISION TREE - CLASSIFCADOR JA TREINADO EM SCRIPTS ANTERIORES USANDO GRIDSEARCH E VALIDACAO CRUZADA #

####isso vai diminuir o tempo de comptuacao para a analise de valores shap (em datasets grandes)
X_summary = shap.kmeans(X, 100)

### explain all the predictions in the test set
explainer_arvore = shap.KernelExplainer(random.predict_proba, X_summary)
shap_values = explainer_arvore.shap_values(X)

# summarize all prediction's explanations
shap.summary_plot(shap_values, X, features) # In bars

shap.summary_plot(shap_values[0], X, show=False, max_display=X.shape[0], feature_names= features)

shap.summary_plot(shap_values[0], X, feature_names=features)# for AD # Check explanation above in classifier evaluation
shap.summary_plot(shap_values[1], X, feature_names=features)# for CN

#Dependence plot:
shap.dependence_plot("GP", shap_values[0], X, features)

shap.dependence_plot("GLY", shap_values[0], X, features)


# visualize the first prediction's explanation with a force plot
shap.force_plot(explainer_arvore.expected_value[0], shap_values[0][0], features = features)
#algum erro que nao gera figura


##########################################################################################
##########################               SVM         #############################################
##########################################################################################

#explain all the predictions in the test set
X_summary = shap.kmeans(X, 100)

#suggestion given in the kernell:
#from sklearn.pipeline import make_pipeline
#from sklearn.linear_model import LassoLarsIC
#svm = make_pipeline(StandardScaler(with_mean=False), LassoLarsIC())
#svm.fit(X, y)

explainer_svm = shap.KernelExplainer(svm.predict_proba, X_summary)
shap_values = explainer_svm.shap_values(X)

#visualiza todos os features:

shap.summary_plot(shap_values, X, features) # com nome dos features
shap.summary_plot(shap_values[0], X, feature_names=features)# for AD # Check explanation above in classifier evaluation
shap.summary_plot(shap_values[0], X, show=False, max_display=X.shape[0], feature_names= features) # aqui mostra todos os fetures

#Dependence plot:
shap.dependence_plot("GP", shap_values[0], X, features)
shap.dependence_plot("GLOL", shap_values[0], X, features)
shap.dependence_plot("TYR", shap_values[0], X, features)
shap.dependence_plot("CREA", shap_values[0], X, features)

shap.plots.heatmap(shap_values)

# visualize the first prediction's explanation with a force plot
shap.force_plot(explainer_svm.expected_value[0], shap_values[0][0], features = features)

##########################################################################################
##########################     MLP       #############################################
##########################################################################################

####isso vai diminuir o tempo de comptuacao para a analise de valores shap (em datasets grandes)
X_summary = shap.kmeans(X, 100) # pode aumenar o valor apra melhorar o coverage 

#or use:
X_summary = shap.sample(X, 100) # it gives me slighlty differences results in top 5 features

# explain all the predictions in the test set
explainer_rede_neural = shap.KernelExplainer(rede_neural.predict_proba, X_summary)
shap_values = explainer_rede_neural.shap_values(X)

#visualiza todos os features:
#https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html
    
shap.summary_plot(shap_values, X, features) # In bars

shap.summary_plot(shap_values[0], X, show=False, max_display=X.shape[0], feature_names= features)

shap.summary_plot(shap_values[0], X, feature_names=features)# for AD # Check explanation above in classifier evaluation
shap.summary_plot(shap_values[1], X, feature_names=features)# for CN

#Deoendence plot:
shap.dependence_plot("GP", shap_values[0], X, features)
shap.dependence_plot("GLOL", shap_values[0], X, features)
shap.dependence_plot("LA.FA", shap_values[0], X, features)
shap.dependence_plot("CREA", shap_values[0], X, features)




















