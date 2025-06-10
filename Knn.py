# %% [markdown]
# 

# %%
# importações necessárias para o código
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import plotly.express as px

# %%
# Carrega o conjunto de dados 'credit' pré-processado de um arquivo pickle.
# Este arquivo contém os conjuntos de treinamento e teste para as features (X) e o target (y)
with open('credit.pkl', 'rb') as f:
    X_treinamento, y_treinamento, X_teste, y_teste = pickle.load(f)

# %%
# Inicializa o classificador K-Nearest Neighbors (KNN) para o dataset de crédito.
# n_neighbors=5: Define o número de vizinhos a considerar para a classificação.
# metric='minkowski', p=2: Especifica a métrica de distância Euclidiana (p=2 para Minkowski).
knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p= 2)
knn_credit.fit(X_treinamento, y_treinamento)

# %%
# Realiza previsões no conjunto de teste usando o modelo KNN treinado.
previsoes = knn_credit.predict(X_teste)
# Exibe as previsões
previsoes

# %%
# Exibe os valores reais do target do conjunto de teste para comparação.
y_teste

# %%
# Calcula e exibe a acurácia do modelo.
# Isso compara os valores previstos com os valores reais do teste.
accuracy_score(y_teste, previsoes)

# %%
# Cria uma visualização da Matriz de Confusão para o modelo KNN.
cm = ConfusionMatrix(knn_credit)
# Ajusta a matriz de confusão aos dados de treinamento.
cm.fit(X_treinamento, y_treinamento)
# Pontua o modelo e gera o gráfico da matriz de confusão nos dados de teste.
# Isso representa visualmente os verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.
cm.score(X_teste, y_teste)


# %%
# Imprime um relatório de classificação detalhado.
# Isso inclui precisão, recall, f1-score e suporte para cada classe.
print(classification_report(y_teste, previsoes))

# %%
# Carrega o conjunto de dados 'census' pré-processado de um arquivo pickle.
# Assim como o dataset de crédito, este contém conjuntos de treinamento e teste.
with open('census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

# %%
# Inicializa o classificador K-Nearest Neighbors (KNN) para o dataset census.
# n_neighbors=13: Define o número de vizinhos. Valor escolhido por ter uma melhor otimização.
# metric='minkowski', p=2: Utiliza a distância Euclidiana.
knn_census = KNeighborsClassifier(n_neighbors= 13, metric='minkowski', p= 2)
# Treina o modelo KNN usando os dados de treinamento do census.
knn_census.fit(X_census_treinamento, y_census_treinamento)

# %%
# Realiza previsões no conjunto de teste do census usando o modelo KNN treinado.
previsoes = knn_census.predict(X_census_teste)
# Exibe as previsões
previsoes

# %%
# Exibe os valores reais do target do conjunto de teste do census.
y_census_teste

# %%
# Calcula e exibe a acurácia para o modelo census.
accuracy_score(y_census_teste, previsoes)

# %%
# Cria e exibe uma visualização da Matriz de Confusão para o modelo KNN do census.
cm = ConfusionMatrix(knn_census)
# Ajusta a matriz de confusão aos dados de treinamento.
cm.fit(X_census_treinamento, y_census_treinamento)
# Pontua o modelo e gera o gráfico da matriz de confusão nos dados de teste.
# Isso representa visualmente os verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.
cm.score(X_census_teste, y_census_teste)

# %%
# Imprime um relatório de classificação detalhado para o modelo census.
print(classification_report(y_census_teste, previsoes))


