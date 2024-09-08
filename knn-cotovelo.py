import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregar os dados
data = pd.read_csv('dados.csv')

# Selecionar os atributos para clusterização
X = data[['Altura', 'Peso', 'IMC']]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar o método do cotovelo
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotar o gráfico do método do cotovelo
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inertia')
plt.title('Método do Cotovelo para Determinar k')
plt.show()

# Escolher o número de clusters com base no gráfico e aplicar k-Means
optimal_k = 3  # Exemplo, escolha o valor baseado no gráfico
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizar os dados clusterizados
print(data.head())