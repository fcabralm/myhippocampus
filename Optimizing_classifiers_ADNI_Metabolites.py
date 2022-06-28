#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:27:48 2022

@author: felipecabralmiranda
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

metabo_adni = pd.read_csv('/Users/felipecabralmiranda/Desktop/Final2.csv')
metabo_adni = metabo_adni.replace(['NDEF'],0)
metabo_adni = metabo_adni.replace(['TAG'],0)
metabo_adni = metabo_adni.fillna(0)

X_adni = metabo_adni.iloc[:, 1:228].values
y_adni = metabo_adni.iloc[:, 228].values

from sklearn.preprocessing import StandardScaler
scaler_adni = StandardScaler()
X_adni = scaler_adni.fit_transform(X_adni)

#carregar classificadores ja treinados no banco de dados (metodo: gridsearch e validacao cruzada)
import pickle
rede_neural = pickle.load(open('rede_neural_finalizado.sav', 'rb'))
arvore = pickle.load(open('arvore_finalizado.sav', 'rb'))
svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random = pickle.load(open('random_forest_adni.sav', 'rb')) 

#example:
exemplo = X_adni[701] # a linha saiu em forma de coluna !
exemplo = exemplo.reshape(1,-1)

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



