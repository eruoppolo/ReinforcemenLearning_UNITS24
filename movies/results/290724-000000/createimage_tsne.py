import numpy as np
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

with open('../../../final/users_data.json', 'r') as f:
    data = json.load(f)


# Generate plots for multiple users
#u = np.random.choice(range(602), size=30, replace=False)
u = [0, 272, 539, 260, 506, 314, 400, 214, 200, 27, 215, 363, 438, 8, 558, 554, 224, 146, 153, 219, 142, 208, 585, 140, 381, 418, 182, 63, 521, 479, 452]
d = len(u)
for i in u:
    user = data[i]
    user_embedding = np.array(user['usr'])
    candidate_items = np.array(user['items'])
    recommended_items = np.array(user['rec_items'])

    # Combine candidate and recommended items
    all_items = np.vstack((candidate_items, recommended_items))

    # t-SNE for 3D visualization
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=min(30, all_items.shape[0] - 1))
    reduced_data_3d = tsne_3d.fit_transform(all_items)

    # Separate the reduced data back into candidate and recommended
    reduced_data = reduced_data_3d[:len(candidate_items)]
    reduced_selected = reduced_data_3d[len(candidate_items):]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter_all = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                             c='blue', alpha=0.6, label='Candidati')
    scatter_selected = ax.scatter(reduced_selected[:, 0], reduced_selected[:, 1], reduced_selected[:, 2],
                                  c='red', s=100, label='Raccomandati')

    ax.set_xlabel('Prima Componente t-SNE')
    ax.set_ylabel('Seconda Componente t-SNE')
    ax.set_zlabel('Terza Componente t-SNE')

    plt.title('t-SNE 3D Plot: Candidati vs Raccomandati')
    plt.legend()
    plt.savefig(f'scatterTSNE/plot3D_tsne_user{i}.png')

    # t-SNE for 2D visualization
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, all_items.shape[0] - 1))
    reduced_data_2d = tsne_2d.fit_transform(all_items)

    # Separate the reduced data back into candidate and recommended
    reduced_data = reduced_data_2d[:len(candidate_items)]
    reduced_selected = reduced_data_2d[len(candidate_items):]

    # Create 2D plot
    plt.figure(figsize=(12, 10))

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                c='blue', alpha=0.6, label='Candidati')
    plt.scatter(reduced_selected[:, 0], reduced_selected[:, 1],
                c='red', s=100, label='Raccomandati')

    plt.xlabel('Prima Componente t-SNE')
    plt.ylabel('Seconda Componente t-SNE')

    plt.title('t-SNE 2D Plot: Candidati vs Raccomandati')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'scatterTSNE/plot2D_tsne_user{i}.png')

    d -= 1
    print(f'----------- USER {i} TERMINATO, RESTANTI {d}')
