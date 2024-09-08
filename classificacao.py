import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# Carregar os dados
data = pd.read_csv('dados.csv')

# Balanceamento das classes: Upsample das classes minoritárias
def balancear_classes(data):
    classes = data['Classificacao'].unique()
    maior_classe = data['Classificacao'].value_counts().idxmax()
    maior_tamanho = data['Classificacao'].value_counts().max()
    
    frames = []
    for classe in classes:
        subset = data[data['Classificacao'] == classe]
        if classe != maior_classe:
            subset_upsampled = resample(subset, replace=True, n_samples=maior_tamanho, random_state=42)
            frames.append(subset_upsampled)
        else:
            frames.append(subset)
    return pd.concat(frames)

data_balanced = balancear_classes(data)

# Selecionar os atributos e o rótulo
X = data_balanced[['Peso', 'Altura']]
y = data_balanced['Classificacao']

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar e treinar o modelo k-NN com valor ajustado de k
knn = KNeighborsClassifier(n_neighbors=5)  # Ajuste do valor de k
knn.fit(X_train_scaled, y_train)

# Fazer previsões com k-NN
y_pred_knn = knn.predict(X_test_scaled)

# Avaliar o modelo k-NN
print("Modelo k-NN:")
print("Acurácia:", accuracy_score(y_test, y_pred_knn))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_knn))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred_knn))

# Criar e treinar o modelo de Árvore de Decisão
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Fazer previsões com Árvore de Decisão
y_pred_tree = tree.predict(X_test)

# Avaliar o modelo de Árvore de Decisão
print("\nModelo de Árvore de Decisão:")
print("Acurácia:", accuracy_score(y_test, y_pred_tree))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_tree))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred_tree))