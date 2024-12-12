import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import preprocessing
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# ###### WIP
# pocket_coords = pocket_data[2]
# all_pockets = []
# for coord_string in pocket_coords:
#     one_pocket = []
#     temp = coord_string[0].split('|')
#     for coord in temp:
#         one_pocket.append(coord.split(','))
#     all_pockets.append(one_pocket)
# all_pockets = np.array(all_pockets)
# all_pockets = all_pockets.astype(float)

X, Xsc, X_pocket, Xsc_pocket, Y, Ysc, Y_pocket, Ysc_pocket = preprocessing.main()

print(Xsc.shape)
print(Xsc_pocket.shape)

def dimReduce(data, Ysc, lab = None):
    # fit it using PCA
    endlab = []
    pca = PCA(n_components = 4)
    pca.fit(data)

    data_transform = pca.transform(data)

    for i, comp in enumerate(pca.explained_variance_ratio_):
        print("The variance explained by principal component {} is {:>5.2f}%".format(i + 1, comp*100))
    print("Total variance explained by this subset is {:>5.2f}%".format(100*np.sum(pca.explained_variance_ratio_)))

    # plot all of the dimensionality reduction methods
    fig, axes = plt.subplots(1, 3, figsize = (24, 5))

    # plot the PCA reduced data
    ax = axes[0]
    ax.scatter(data_transform[:, 0], data_transform[:, 1], marker='.', cmap='viridis', c=Ysc)
    ax.set_title("PCA")

    # plot the tsne reduced data
    tsne = TSNE(n_components = 2)
    data_tsne = tsne.fit_transform(data)

    ax = axes[1]
    ax.scatter(data_tsne[:, 0], data_tsne[:, 1], c=Ysc, cmap = 'viridis', marker='.')
    ax.set_title("t-SNE for Ligands")

    # plot the tsne reduced pca data
    data_tsne_pca = tsne.fit_transform(data_transform)

    ax = axes[2]
    ax.scatter(data_tsne_pca[:, 0], data_tsne_pca[:, 1], c=Ysc, cmap = 'viridis', marker='.')
    ax.set_title("t-SNE w/ PCA")

    # reducer = umap.UMAP()
    # data_umap = reducer.fit_transform(data)
    # ax = axes[3]
    # ax.scatter(data_umap[:, 0], data_umap[:, 1], c = Ysc, cmap = 'viridis', marker='.')
    # ax.set_title("UMAP")

    plt.show()


    # now using clustering methods

    # starting with kmeans
    kmax = 19
    fig, axes = plt.subplots(2, 5, figsize = (20, 5), constrained_layout=True)
    axes = axes.flatten()

    for k in range(10, kmax + 1):
        kmeans = KMeans(n_clusters = k, random_state = 0).fit(data_tsne_pca)
        if (lab is None):
            labels = kmeans.labels_
        else:
            labels = lab
        axes[k - 10].scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap = 'viridis', marker='.')
    plt.show()


    # clustering with dbscan
    epsilon = [4, 5]
    n_min = [1, 3]

    fig, axes = plt.subplots(len(n_min), len(epsilon), figsize = (20, 20), constrained_layout=True)
    for i in range(len(epsilon)):
        for j in range(len(n_min)):
            dbscan = DBSCAN(eps = epsilon[i], min_samples = n_min[j]).fit(data_tsne_pca)
            if (lab is None):
                labels = dbscan.labels_
                if (i == len(epsilon) - 1):
                    endlab = labels
            else:
                labels = lab
            ax = axes[j][i]
            ax.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap = 'viridis', marker='.')
            ax.set_title(f"Eps = {epsilon[i]}")
            ax.set_ylabel(f"N_min = {n_min[j]}")
    plt.show()
    
    return data_tsne_pca, endlab

pocket_data_tsne, endlab = dimReduce(X_pocket, Y_pocket)
ligand_data_tsne, temp = dimReduce(X, Y, endlab)

def EuclideanDistance(x1, y1, x2, y2):
    dist = pow(x1 - x2, 2) + pow(y1 - y2, 2)
    return math.sqrt(dist)

def pairwiseSimilarity(data):
    similarity_matrix = np.zeros((len(data), len(data)))
    similarity = []
    for i in range(0, len(data) - 1):
        for j in range(i + 1, len(data)):
            x1 = data[i][0]
            y1 = data[i][1]
            x2 = data[j][0]
            y2 = data[j][1]
            dist = EuclideanDistance(x1, y1, x2, y2)
            similarity_matrix[i][j] = dist
            similarity_matrix[j][i] = dist
            similarity.append(dist)
            
    return similarity_matrix, similarity

ligand_similarity_matrix, ligand_similarity = pairwiseSimilarity(ligand_data_tsne)
pocket_similarity_matrix, pocket_similarity = pairwiseSimilarity(pocket_data_tsne)

diff = ligand_similarity_matrix - pocket_similarity_matrix

# fig, axes = plt.subplots(ncols = 3)
# sns.heatmap(ligand_similarity_matrix, cmap = 'binary', ax = axes[0])
# sns.heatmap(pocket_similarity_matrix, cmap = 'binary', ax = axes[1])
# sns.heatmap(diff, cmap = 'seismic', ax = axes[2])
# fig.suptitle("Pairwise Difference in Similarity Heatmap")
# plt.show()

scaler = StandardScaler()
ligand_similarity = np.array(ligand_similarity).reshape(-1, 1)
scaler.fit(ligand_similarity)
ligand_similarity = scaler.transform(ligand_similarity)

scaler = StandardScaler()
pocket_similarity = np.array(pocket_similarity).reshape(-1, 1)
scaler.fit(pocket_similarity)
pocket_similarity = scaler.transform(pocket_similarity)

fig , axes = plt.subplots(ncols = 3)
axes[0].hist(np.array(ligand_similarity), bins = 100)
axes[1].hist(np.array(pocket_similarity), bins = 100)
axes[2].hist(np.array(ligand_similarity) - np.array(pocket_similarity), bins = 100)

plt.show()

plt.hist2d(np.ravel(ligand_similarity), np.ravel(pocket_similarity), bins = (100, 100), cmap = plt.cm.jet)
plt.xlabel("Ligand Similarity")
plt.ylabel("Protein Pocket Similarity")
plt.show()

#began with a much smaller dataset, it was difficult to characterize and did not cluster well; using a larger dataset allowed for
# more defined clusters to form

# used scaled data, and then used non scaled data
# non scaled data seemed to have a lot more variance explained by pca and seems to cluster groups much more effectively
# in particular, with the non scaled data with pocket proteins, the cluster