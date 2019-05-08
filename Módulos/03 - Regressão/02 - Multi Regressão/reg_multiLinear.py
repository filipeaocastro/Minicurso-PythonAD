"""
Minicurso de Python para Análise de Dados

Filipe Augusto

Multiregressão Linear

"""

# Importando as bibliotecas
import numpy as np                 #Para trabalhar com matrizes
import matplotlib.pyplot as plt    #Para trabalhar com gráficos
import pandas as pd                #Para trabalhar com datasets

#Importing the dataset 
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values    #Cria uma matriz com os valores X (independente)
y = dataset.iloc[:, 4].values      #Cria uma matriz com os valores y (dependente)

# Transformando dados categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Transformando a variável independente
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable trap
X = X[:, 1:] #Tira a primeira coluna de Dummy Variables

#Dividindo o dataset em Training set e Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#REGRESSÃO LINEAR MÚLTIPLA 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predição
y_pred = regressor.predict(X_test)

t = np.arange(0, len(y_pred))
plt.figure(1)
plt.plot(t, y_pred, color = 'red', label = 'y_pred')
plt.plot(t, y_test, color = 'black', label = 'y_test')
plt.legend(loc='upper right')
plt.show()



# ************************** #

# Eliminando variáveis irrelevantes e fazendo Backward Elimination

# Adiciona uma coluna de números 1 e remove as colunas cujo valor p seja maior que 0.05
# 0.05 = nível de sugnificância
# É removida uma coluna por vez, a cada remoção é feita a análise de significância novamente
# O processo só acaba quando todas as colunas possuirem valor p <= 0.05
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Ao remover as colunas irrelevantes o modelo de predição é refeito com a nova matriz de X
X_train_opt, X_test_opt, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, 
                                                            random_state = 0)
regressor_opt = LinearRegression()
regressor_opt.fit(X_train_opt, y_train)

y_pred_opt = regressor_opt.predict(X_test_opt)

t_opt = np.arange(0, len(y_pred_opt), 1)

plt.figure(2)
plt.plot(t_opt, y_test, color = 'red')
plt.plot(t_opt, y_pred_opt, color = 'blue')
plt.show()

