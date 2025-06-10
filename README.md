# 🤖 Projeto KNN - Classificação de Dados com Python

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Yellowbrick](https://img.shields.io/badge/Yellowbrick-FCC200?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjRkZGIiBoZWlnaHQ9IjI0IiB3aWR0aD0iMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiByeD0iNCIgZmlsbD0iI0ZDMjAwMCIvPjwvc3ZnPg==&label)

## 🔍 Este projeto demonstra o uso do algoritmo **K-Nearest Neighbors (KNN)** para classificação de dados em duas bases distintas: uma relacionada a crédito e outra ao censo. Utilizamos bibliotecas populares como `scikit-learn`, `pandas`, `matplotlib`, `plotly` e `yellowbrick` para processamento, modelagem e visualização.

## 📁 Estrutura do Projeto

- `census.pkl`   # Base de dados do censo (treinamento e teste)
- `credit.pkl`   # Base de dados de crédito (treinamento e teste)
- `Knn.ipynb`    # Jupyter Notebook
- `Knn.py`       # Código Python
- `README.md`    # Este arquivo

## 🚀 Tecnologias Utilizadas

- Python
- Scikit-learn
- Yellowbrick
- Pandas / Numpy
- Matplotlib / Plotly

## 📊 Avaliação dos Modelos

O projeto mostra a acurácia e métricas como precisão, recall e f1-score com uso da matriz de confusão e relatórios de classificação.

                precision   recall   f1-score   support

       <=50K       0.87      0.92      0.89      3693
        >50K       0.69      0.56      0.62      1192

    accuracy                           0.83      4885
   macro avg       0.78      0.74      0.76      4885
weighted avg       0.82      0.83      0.82      4885

## 📌 Funcionalidades

- Treinamento com KNN usando métricas de distância Euclidiana
- Avaliação dos modelos
- Visualização de métricas com Yellowbrick

## 👨‍💻 Quem Desenvolveu

Projeto desenvolvido por:

**Daniel Campos Coelho**  
🎓 Estudante de Ciências da Computação  
🔗 [LinkedIn](https://www.linkedin.com/in/daniel-coelho-818381293/)  
💻 [GitHub](https://github.com/daccoelho)

---

## ✨ Contribuições

Contribuições são muito bem-vindas! Sinta-se à vontade para fazer um fork, abrir uma issue ou mandar um pull request.

---
