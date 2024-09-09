import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

dados = pd.read_csv('dados.csv')

atributos = dados[['Altura', 'Peso', 'IMC']]

escalador = StandardScaler()
atributos_normalizados = escalador.fit_transform(atributos)

matriz_ligacao = linkage(atributos_normalizados, method='ward')

numero_clusters = 3
dados['Cluster_Hierarquico'] = fcluster(matriz_ligacao, numero_clusters, criterion='maxclust')

print(dados.head())

dados.to_csv('dados_clusterizados.csv', index=False)