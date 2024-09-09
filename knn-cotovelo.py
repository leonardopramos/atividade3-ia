import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

dados = pd.read_csv('dados.csv')

atributos = dados[['Altura', 'Peso', 'IMC']]

#Serve para padronizar os dados
escalador = StandardScaler()
atributos_padronizados = escalador.fit_transform(atributos)

inercias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(atributos_padronizados)
    inercias.append(kmeans.inertia_)

k_otimo = 3

kmeans_final = KMeans(n_clusters=k_otimo, random_state=42)
dados['Cluster'] = kmeans_final.fit_predict(atributos_padronizados)

print(dados.head())