# Importar bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Carregar os dados
data = pd.read_csv('dados.csv')

# Selecionar os atributos para clusterização
X = data[['Altura', 'Peso', 'IMC']]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar o agrupamento hierárquico
linkage_matrix = linkage(X_scaled, method='ward')

# Plotar o dendrograma
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Dendrograma do Agrupamento Hierárquico')
plt.xlabel('Índice da Amostra')
plt.ylabel('Distância')
plt.show()

# Escolher o número de clusters com base no dendrograma
num_clusters = 3  # Exemplo, escolha o valor baseado no dendrograma
data['Cluster_Hierarquico'] = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Visualizar os dados clusterizados
print(data.head())

# Salvar os resultados em um novo arquivo CSV
data.to_csv('dados_clusterizados.csv', index=False)