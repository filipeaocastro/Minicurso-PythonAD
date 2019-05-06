"""
Minicurso de Python para Análise de Dados

Filipe Augusto

Fundamentos Básicos de Python

"""
# Importando as bibliotecas
import numpy as np                 #Para trabalhar com matrizes
import matplotlib.pyplot as plt    #Para trabalhar com gráficos
import pandas as pd                #Para trabalhar com datasets

# Variáveis e Operadores
a = 2                   # Número inteiro
b = 3.1415              # Ponto flutuante
c = 'python'            # String
bool1 = True            # Booleano Verdadeiro
bool2 = False           # Booleano Falso
nada = None             # Variável Nula
print(bool1 and bool2)  # Operador 'e'
print(bool1 or bool2)   # Operador 'ou'
print(bool1 != bool2)   # Operador 'diferente de'
print(bool1 == bool2)   # Operador 'igual a'


# Funções Básicas
l1 = [1, 4, 6, -55, 98, 26195121]
l2 = [5, 8, 2, 46, -57, -1122222]
a = -20

len(l1)             # Retorna o comprimento da variável
range(6, 20, 2)     # Cria um objeto interável que vai de 6 a 20 de 2 em 2
int(a)              # Transforma x em um número inteiro
str(a)              # Transforma x em uma string
float(a)            # Transforma x em um ponto flutuante
abs(a)              # Retorna o valor absoluto de x
max(l1)             # Retorna o maior valor
min(l1)             # Retorna o menor valor
print(l1)           # Imprime a variável no console
type(l1)            # Retorna o tipo de variável
zip(l1, l2)         # Retorna um interador a partir de um ou mais interadores

# If/Else
a = 5
b = 5

if (b - a) == 0 or (a - b) == 0 :
    print('a e b são iguais')
    
else:
    if (b / a) == a:
        print('a é raiz de b')
    else:
        print('a é diferente de b e não é sua raiz')
        
# Loops

# while
a = 0
b = 0
while (a + b) < 10:
    a += 1
    b += 1
    print('\na = ' + str(a) + '\nb = ' + str(b))
    
# for
for i in range(0, 10, 2):
    print(i)

   
impar = [1, 3, 5, 7, 9]
par = [2, 4, 6, 8]

for i, j in zip(impar, par):
    print(i, j)


# Matrizes Numpy
z = np.zeros((5,3))             # Cria uma matriz de zeros
u = np.ones((6,3))              # Cria uma matriz de uns
np.transpose(z)                 # Transpõe a matriz
m = np.array([[1, 2], [3, 4]])  # Cria uma matriz
ar = np.arange(0, 30, 3)        # Cria um vetor com valores igualmente espaçados, onde você define 
                                #o tamanho do espaço
np.linspace(0, 30, 3)           # Cria um vetor com valores igualmente espaçados, onde você define 
                                #o tamanho do vetor
np.mean(ar)                     # Calcula a média de uma matriz
np.std(ar)                      # Calcula o desvio padrão de uma matriz                  
np.pi                           # Valor pi

# Datasets
df = pd.DataFrame(m, columns = ['Dado 1', 'Dado 2'])    # Cria um data frame
d1 = df.iloc[:, 0].values                               # Pega os valores da primeira coluna
d2 = df.iloc[:, 1].values                               # Pega os valores da segunda coluna

# Gráficos
y_1 = np.linspace(0, 20, 9)
y_2 = np.linspace(20, 0, 9)
x = np.arange(0, len(y_1))
plt.figure(1)                                       # Define o número da figura
plt.plot(x, y_1, color = 'red', label = 'x_1')      # Plota um gráfico de linha
plt.scatter(x, y_2, color = 'blue', label = 'x_2')  # Plota um gráfico de pontos
plt.xlabel("Eixo X")                                # Dá um nome pro Eixo X 
plt.ylabel("Eixo Y")                                # Dá um nome pro Eixo Y
plt.title('Gráfico de exemplo')                     # Dá um título ao gráfico
plt.legend(loc='upper right')                       # Define o local da legenda
plt.show()                                          # Mostra o gráfico