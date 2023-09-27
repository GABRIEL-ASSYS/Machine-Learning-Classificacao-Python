#declaracao das bibliotecas
import panda as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#lendo as informacoes do csv
data = pd.read_excel('ataque_do_coracao.csv', sep=',')
data = data.dropna()

#removendo os espacos vazios e formatando os dados
data['Patient ID'] = data['Patient ID'].str.strip().astype(str)
data['Age'] = data['Age'].str.strip().astype(int)
data['Sex'] = data['Sex'].str.strip().astype(str)
data['Cholesterol'] = data['Cholesterol'].str.strip().astype(int)
data['Blood Pressure'] = data['Blood Pressure'].str.strip().astype(str)
data['Heart Rate'] = data['Heart Rate'].str.strip().astype(int)
data['Diabetes'] = data['Diabetes'].str.strip().astype(bool)
data['Family History'] = data['Family History'].str.strip().astype(bool)
data['Smoking'] = data['Smoking'].str.strip().astype(bool)
data['Obesity'] = data['Obesity'].str.strip().astype(bool)

#define X e Y
x = data.drop('Cholesterol', axis=1)
y = data['Age']
print(data.info())

#dividindo em porcao de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Regrecao logistica
lr = LogisticRegression() #cria a variavel
lr.fit(x_train, y_train) #cria o modelo baseado nos valores de treino de x e y
lr_pred = lr.predict(x_test) #faz a previsao com os valores de treino de x
lr_accuracy = accuracy_score(y_test, lr_pred) #calcula acuracia
print(f'Acuracia da Regressão Logística: {lr_accuracy}') #mostra valor da acuracia

#Arvore de Decisao
dt = DecisionTreeClassifier() #cria a variavel
dt.fit(x_train, y_train) #cria o modelo baseado nos valores de treino de x e y
dt_pred = dt.predict(x_test) #faz a previsao com os valores de treino de x
dt_accuracy = accuracy_score(y_test, dt_pred) #calcula acuracia
print(f'Acurácia da Árvore de decisão: {dt_accuracy}') #mostra valor da acuracia

#Random Forest
rf = RandomForestClassifier() #cria a variavel
rf.fit(x_train, y_train) #cria o modelo baseado nos valores de treino de x e y
rf_pred = rf.predict(x_test) #faz a previsao com os valores de treino de x
rf_accuracy = accuracy_score(y_test, rf_pred) #calcula acuracia
print(f'Acurácia da Random Forest: {rf_accuracy}') #mostra valor da acuracia

#SVM
svm = SVC() #cria a variavel
svm.fit(x_train, y_train) #cria o modelo baseado nos valores de treino de x e y
svm_pred = svm.predict(x_test) #faz a previsao com os valores de treino de x
svm_accuracy = accuracy_score(y_test, svm_pred) #calcula acuracia
print(f'Acurácia da SVM: {svm_accuracy}') #mostra valor da acuracia

#K-Nearest Neighbors (K-NN)
knn = KNeighborsClassifier() #cria a variavel
knn.fit(x_train, y_train) #cria o modelo baseado nos valores de treino de x e y
knn_pred = knn.predict(x_test) #faz a previsao com os valores de treino de x
knn_accuracy = accuracy_score(y_test, knn_pred) #calcula acuracia
print(f'Acurácia da KNN: {knn_accuracy}') #mostra valor da acuracia

