# users_data.json


import numpy as np
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

with open('../../../movies/users_data.json', 'r') as f:
    data = json.load(f)
u = 0
print(f'Len data = {len(data)}')
first_user = data[u]
user_embedding = np.array(first_user['usr'])
candidate_items = np.array(first_user['items'])
recommended_items = np.array(first_user['rec_items'])

pca = PCA(n_components=3)
reduced_data = pca.fit_transform(candidate_items)
reduced_selected = pca.transform(recommended_items)

# Crea il grafico 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotta tutti i punti in blu
scatter_all = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                         c='blue', alpha=0.6, label='Candidati')

# Plotta i punti selezionati in rosso
scatter_selected = ax.scatter(reduced_selected[:, 0], reduced_selected[:, 1], reduced_selected[:, 2],
                              c='red', s=100, label='Raccomandati')

# Etichette degli assi
ax.set_xlabel('Prima Componente Principale')
ax.set_ylabel('Seconda Componente Principale')
ax.set_zlabel('Terza Componente Principale')

plt.title('PCA 3D Plot: Candidati vs Raccomandati')
plt.legend()
plt.savefig(f'plot3D_user{u}.png')

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(candidate_items)
reduced_selected = pca.transform(recommended_items)

# Crea il grafico 2D
plt.figure(figsize=(12, 10))

# Plotta tutti i punti in blu
plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
            c='blue', alpha=0.6, label='Candidati')

# Plotta i punti selezionati in rosso
plt.scatter(reduced_selected[:, 0], reduced_selected[:, 1],
            c='red', s=100, label='Raccomandati')

# Etichette degli assi
plt.xlabel('Prima Componente Principale')
plt.ylabel('Seconda Componente Principale')

plt.title('PCA 2D Plot: Candidati vs Raccomandati')
plt.legend()

# Aggiungi una griglia per una migliore leggibilità
plt.grid(True, linestyle='--', alpha=0.7)

# Aggiungi annotazioni per mostrare la varianza spiegata
# var_ratio = pca.explained_variance_ratio_
# plt.annotate(f'Varianza spiegata: {var_ratio[0]:.2f}, {var_ratio[1]:.2f}',
#              xy=(0.05, 0.95), xycoords='axes fraction')

plt.tight_layout()

plt.savefig(f'plot2D_user{u}.png')


u = np.random.choice(range(602), size=30, replace=False)
d = len(u)
for i in u:
    user = data[i]
    user_embedding = np.array(user['usr'])
    candidate_items = np.array(user['items'])
    recommended_items = np.array(user['rec_items'])

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(candidate_items)
    reduced_selected = pca.transform(recommended_items)

    # Crea il grafico 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plotta tutti i punti in blu
    scatter_all = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                            c='blue', alpha=0.6, label='Candidati')

    # Plotta i punti selezionati in rosso
    scatter_selected = ax.scatter(reduced_selected[:, 0], reduced_selected[:, 1], reduced_selected[:, 2],
                                c='red', s=100, label='Raccomandati')

    # Etichette degli assi
    ax.set_xlabel('Prima Componente Principale')
    ax.set_ylabel('Seconda Componente Principale')
    ax.set_zlabel('Terza Componente Principale')

    plt.title('PCA 3D Plot: Candidati vs Raccomandati')
    plt.legend()
    plt.savefig(f'scatterPCA/plot3D_user{i}.png')


    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(candidate_items)
    reduced_selected = pca.transform(recommended_items)

    # Crea il grafico 2D
    plt.figure(figsize=(12, 10))

    # Plotta tutti i punti in blu
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                c='blue', alpha=0.6, label='Candidati')

    # Plotta i punti selezionati in rosso
    plt.scatter(reduced_selected[:, 0], reduced_selected[:, 1],
                c='red', s=100, label='Raccomandati')

    # Etichette degli assi
    plt.xlabel('Prima Componente Principale')
    plt.ylabel('Seconda Componente Principale')

    plt.title('PCA 2D Plot: Candidati vs Raccomandati')
    plt.legend()

    # Aggiungi una griglia per una migliore leggibilità
    plt.grid(True, linestyle='--', alpha=0.7)

    # Aggiungi annotazioni per mostrare la varianza spiegata
    # var_ratio = pca.explained_variance_ratio_
    # plt.annotate(f'Varianza spiegata: {var_ratio[0]:.2f}, {var_ratio[1]:.2f}',
    #              xy=(0.05, 0.95), xycoords='axes fraction')

    plt.tight_layout()

    plt.savefig(f'scatterPCA/plot2D_user{i}.png')
    
    d-=1

    print(f'----------- USER {i} TERMINATO, RESTANTI {d}')
