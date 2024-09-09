import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

dados = pd.read_csv('dados.csv')

def balancear(dados):
    tipos = dados['Classificacao'].unique()
    tipo_mais_comum = dados['Classificacao'].value_counts().idxmax()
    tamanho_maior = dados['Classificacao'].value_counts().max()
    
    lista_dados = []
    for tipo in tipos:
        grupo = dados[dados['Classificacao'] == tipo]
        if tipo != tipo_mais_comum:
            grupo_ajustado = resample(grupo, replace=True, n_samples=tamanho_maior, random_state=42)
            lista_dados.append(grupo_ajustado)
        else:
            lista_dados.append(grupo)
    return pd.concat(lista_dados)

dados_ajustados = balancear(dados)

atributos = dados_ajustados[['Peso', 'Altura']]
classes = dados_ajustados['Classificacao']

#Divisao teste e treino
atributos_treino, atributos_teste, treino_classes, teste_classes = train_test_split(atributos, classes, test_size=0.3, random_state=42)

normalizador = StandardScaler()
atributos_treino_norm = normalizador.fit_transform(atributos_treino)
atributos_teste_norm = normalizador.transform(atributos_teste)

modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(atributos_treino_norm, treino_classes)

previsoes = modelo_knn.predict(atributos_teste_norm)

print("Acurácia do knn:", accuracy_score(teste_classes, previsoes))