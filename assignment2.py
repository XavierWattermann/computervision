import pdb
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from Code.AutoencoderPyTorchCode.simpleAutoencoder import AE
from sklearn.manifold import TSNE, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# import torch
# import torch.utils.data as Data


# load data
# D = np.loadtxt("Dataset/MNIST/MNIST_digits0-1-2.csv", delimiter=",")
# breakpoint()

# =================== READ IN MNIST / FMNIST ================================
# DataSet needs to be in the same "level" as assignment1.py (this file)
path_location = Path(__file__).resolve().parent
data_set_path = file_path = path_location / "DataSet"
# TODO: make this a cli variable to pass in?
mnist = False  # change this (to False) to use FMNIST dataset
digits_path = "Digits.p" if mnist else "FMNISTTestData_Org.p"
labels_path = "Labels.p" if mnist else "FMNISTTestLabel_Org.p"

print("DEBUG: ", mnist, digits_path, labels_path)

with open(data_set_path / digits_path, "rb") as openfile:
    digits = pickle.load(openfile)
with open(data_set_path / labels_path, "rb") as openfile:
    labels = pickle.load(openfile)


def get_scatter(model, ax, labels):
    """
    helper function to generate the scatter plt

    model -> the model (lle, le, t-sne) fit_transform
    ax -> matplotlib axis, the specific row/column we're dealing with for the overall figure
    labels -> from the labels.p file (this could probably be a global?)
    """

    scatter = ax.scatter(
        model[:, 0], model[:, 1], c=labels, cmap="tab10", s=1, alpha=0.5
    )
    return scatter


def create_legend(fig, handles, mnist=True, loc="center right"):
    """
    Creates the legend for mnist or fmnist (passed in as a var)
    so either displays the digits or the labels of the fashion items

    source for the fashion items was found here:
    https://github.com/fendy07/FMNIST-DeepLearning/tree/master?tab=readme-ov-file

    fig -> the matplotlib figure to adjust
    handles -> comes from the scatter plot
    loc -> where to put the legend
    mnist
    """
    fmnist_map = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    if mnist:
        fig.legend(
            handles,
            [str(i) for i in range(10)],
            loc=loc,
            title="Digits",
            bbox_to_anchor=(1.0, 0.5),  # adjust this?
        )
    else:
        fig.legend(
            handles,
            [fmnist_map[(i)] for i in range(10)],
            loc=loc,
            title="Fashion Item",
            bbox_to_anchor=(1.0, 0.5),
        )


# ========================= PART II ==================================
test_amount = 200
train_index = []
test_index = []
scores = []
for digit in np.unique(labels):
    index = np.where(labels == digit)[0]

    test_index.append(index[:test_amount])
    train_index.append(index[test_amount:])

# we have a list of lists, need to merge them into one
train_index = [index for _list in train_index for index in _list]
test_index = [index for _list in test_index for index in _list]
x_train_mnist, l_train_mnist = digits[train_index], labels[train_index]
x_test_mnist, l_test_mnist = digits[test_index], labels[test_index]


# TODO: need to change legend
fig, axes = plt.subplots(5, 2, figsize=(14, 18))
scatter = None
for count in range(5):
    model = AE()
    model.fit(
        trData=x_train_mnist,
        lrnRate=0.001,
        nEpochs=200,
        miniBatchSize=64,
        cudaDeviceId=0,
    )
    model = model.to("cpu")
    # breakpoint()
    torch_data_train = torch.Tensor(torch.from_numpy(x_train_mnist).float())
    encoded_data_train = model.encoder(torch_data_train).detach().numpy()

    torch_data_test = torch.Tensor(torch.from_numpy(x_test_mnist).float())
    encoded_data_test = model.encoder(torch_data_test).detach().numpy()

    # 5NN Accuracy
    k_nn_amount = 5
    knn_classifer = KNeighborsClassifier(n_neighbors=k_nn_amount)
    knn_classifer.fit(encoded_data_train, l_train_mnist.ravel())

    # train_score = knn_classifer.score(encoded_data_train, l_train_mnist)
    test_score = knn_classifer.score(encoded_data_test, l_test_mnist)

    # print(train_score, test_score)
    # print(test_score)
    scores.append(test_score)

    # PLOT
    # TODO: make a function for this; make axes[count,0] (and ,1) into a variable
    scatter_train = axes[count, 0].scatter(
        encoded_data_train[:, 0],
        encoded_data_train[:, 1],
        c=l_train_mnist,
        cmap="tab10",
        s=1,
        alpha=0.5,
    )
    axes[count, 0].set_ylabel(f"Trial # {count+6}")
    axes[count, 0].set_xticks([])
    axes[count, 0].set_yticks([])
    if count == 0:
        axes[count, 0].set_title("Training")
    scatter = axes[count, 1].scatter(
        encoded_data_test[:, 0],
        encoded_data_test[:, 1],
        c=l_test_mnist,
        cmap="tab10",
        s=1,
        alpha=0.5,
    )
    if count == 0:
        axes[count, 1].set_title("Test")
    axes[count, 1].set_xticks([])
    axes[count, 1].set_yticks([])

handles, _ = scatter.legend_elements(prop="colors", alpha=1.0)
legend = create_legend(fig, handles=handles, mnist=mnist)

plt.suptitle(f"{'MNIST' if mnist else 'FMNIST'} Autoencoder Train & Test")

# plt.tight_layout()
print(scores)
# plt.savefig("pdfTest.pdf")
plt.savefig(f"{'MNIST' if mnist else 'FMNIST'} Autoencoder Train & Test - 6-10.pdf")

# ========================= PART II END ===============================


# ========================= PART II - QUESTION II START ===============

# for count in range(10):
#     print(f"On Trial: {count}")
#     model = AE()
#     model.fit(
#         trData=x_train_mnist,
#         lrnRate=0.001,
#         nEpochs=200,
#         miniBatchSize=64,
#         cudaDeviceId=0,
#     )
#     model = model.to("cpu")
#     torch_data_train = torch.Tensor(torch.from_numpy(x_train_mnist).float())
#     encoded_data_train = model.encoder(torch_data_train).detach().numpy()

#     torch_data_test = torch.Tensor(torch.from_numpy(x_test_mnist).float())
#     encoded_data_test = model.encoder(torch_data_test).detach().numpy()

#     # 5NN Accuracy
#     k_nn_amount = 5
#     knn_classifer = KNeighborsClassifier(n_neighbors=k_nn_amount)
#     knn_classifer.fit(encoded_data_train, l_train_mnist.ravel())

#     train_score = knn_classifer.score(encoded_data_train, l_train_mnist)
#     test_score = knn_classifer.score(encoded_data_test, l_test_mnist)

#     print(train_score, test_score)
#     # print(test_score)
#     scores.append(test_score)


# print(scores)
# for score in scores:
#     print(score)


# ============================ QUESTION 1 -- main plots ===================================
# # Multiple Figure Setup -- Need to tweak a few of these settings
# fig, axes = plt.subplots(3, 5, figsize=(22, 10))
# for ax, row_name in zip(axes[:, 0], ["LLE", "LE", "t-SNE"]):
#     ax.set_ylabel(row_name, rotation=0, size="large", labelpad=30)

# # ************************* LOCALLY LINEAR EMBEDDING (LLE) ******************
# n_neighbors is the hyperparameter for LLE, components == 2d, don't need 3d for this assignment

# use this to set how many times we run it
# trial_count = 5
# print("Starting LLE")
# lle = LocallyLinearEmbedding(n_neighbors=5, n_components=2)
# for count in range(trial_count):

#     lle_transform = lle.fit_transform(digits)

#     ax = axes[0, count]
#     scatter = get_scatter(lle_transform, ax, labels)

#     ax.set_title(f"Run {count+1}")
#     ax.set_xticks([])
#     ax.set_yticks([])

# print("\tEnding LLE")

# # ************************* LAPLACIAN EIGENMAPS (LE) ********************


# print("Starting LE")
# le = SpectralEmbedding(n_neighbors=5, n_components=2)
# for count in range(trial_count):

#     le_transform = le.fit_transform(digits)
#     ax = axes[1, count]
#     scatter = get_scatter(le_transform, ax, labels)

#     ax.set_title(f"Run {count+1}")
#     ax.set_xticks([])
#     ax.set_yticks([])


# print("\tEnding LE")
# # ********************** t-SNE ********************


# print("Starting t-SNE")
# for count in range(trial_count):
#     # look into the init function, default is pca? there's a random option?
#     tsne = TSNE(n_components=2, perplexity=30)
#     tsne_transform = tsne.fit_transform(digits)
#     ax = axes[2, count]
#     scatter = get_scatter(tsne_transform, ax, labels)

#     ax.set_title(f"Run {count+1}")
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # Legend -> add this at the end because i need access to the scatter
#     handles, _ = scatter.legend_elements(prop="colors", alpha=1.0)
#     create_legend(fig, handles, mnist)
#     fig.suptitle(f"LLE, LE, t-SNE on {'MNIST' if mnist else 'FMNIST'} Data Set")

# print("\nEnding t-SNE")

# # plt.show()
# # plt.savefig("part1_figures_mnist.pdf")
# plt.savefig("part1_figures_fmnist.pdf", bbox_inches="tight")

# =============HyperParameters for each method =============

# =============LLE n-neighbors chosen          =============

# subset_size = 500
# subset = np.random.choice(digits.shape[0], size=subset_size)
# subset_digits = digits[subset]
# subset_labels = labels[subset]

test_amount = 200
test_index = []
scores = []
for digit in np.unique(labels):
    index = np.where(labels == digit)[0]
    test_index.append(index[:test_amount])

# we have a list of lists, need to merge them into one
test_index = [index for _list in test_index for index in _list]
subset_digits, subset_labels = digits[test_index], labels[test_index]
subset_size = len(subset_digits)


# print("DEBUG: Starting LLE -- MNIST", mnist)
# # starting value, and how much we want to increase by
# n_neighbor_count = 2
# n_neighbor_increase = 20
# fig, axes = plt.subplots(2, 5, figsize=(18, 4), squeeze=False)
# for ax, row_name in zip(axes[:, 0], ["LLE", "LLE"]):
#     ax.set_ylabel(row_name, rotation=0, size="large", labelpad=30)

# for row in [0, 1]:
#     for count in range(5):
#         print("Count", count)
#         lle = LocallyLinearEmbedding(n_neighbors=n_neighbor_count, n_components=2)
#         lle_transform = lle.fit_transform(subset_digits)

#         ax = axes[row, count]
#         scatter = get_scatter(lle_transform, ax, subset_labels)

#         ax.set_title(f"n = {n_neighbor_count}")
#         ax.set_xticks([])
#         ax.set_yticks([])

#         handles, _ = scatter.legend_elements(prop="colors", alpha=1.0)
#         create_legend(fig, handles, mnist=mnist)

#         # increase the n_neighbor_count for each run
#         n_neighbor_count += n_neighbor_increase
#         fig.suptitle(
#             f"Hyperparameter Analysis - {'MNIST' if mnist else 'FMNIST'} LLE -  Sample Count: {subset_size}, n_neighbor increase={n_neighbor_increase}"
#         )
# plt.show()
# plt.savefig(f"{'mnist' if mnist else 'fmnist'}_lle_large.pdf")
# ===================================================================================================================


# ============= LE n-neighbors chosen          =============

# subset_size = 500
# subset = np.random.choice(digits.shape[0], size=subset_size)
# subset_digits = digits[subset]
# subset_labels = labels[subset]


# print("Starting LE -- MNIST vs FMNIST", mnist)
# # starting value, and how much we want to increase by
# n_neighbor_count = 2
# n_neighbor_increase = 3
# fig, axes = plt.subplots(2, 5, figsize=(18, 4), squeeze=False)
# for ax, row_name in zip(axes[:, 0], ["LE", "LE"]):
#     ax.set_ylabel(row_name, rotation=0, size="large", labelpad=30)

# for row in [0, 1]:
#     for count in range(5):
#         print("Count", count)
#         le = SpectralEmbedding(n_neighbors=n_neighbor_count, n_components=2)
#         le_transform = le.fit_transform(subset_digits)

#         ax = axes[row, count]
#         scatter = get_scatter(le_transform, ax, subset_labels)

#         ax.set_title(f"n = {n_neighbor_count}")
#         ax.set_xticks([])
#         ax.set_yticks([])

#         handles, _ = scatter.legend_elements(prop="colors", alpha=1.0)
#         create_legend(fig, handles, mnist=mnist)

#         # increase the n_neighbor_count
#         n_neighbor_count += n_neighbor_increase
#         fig.suptitle(
#             f"Hyperparameter Analysis - {'MNIST' if mnist else 'FMNIST'} LE -  Sample Count: {subset_size}, n_neighbor increase={n_neighbor_increase}"
#         )
# # plt.show()
# # plt.savefig(f"{'mnist' if mnist else 'fmnist'}_le.pdf")
# ===================================================================================================================


# ============= t-SNE perplexity chosen                   =============

# subset_size = 2000
# subset = np.random.choice(digits.shape[0], size=subset_size)
# subset_digits = digits[subset]
# subset_labels = labels[subset]


# print("Starting tsne -- MNIST vs FMNIST", mnist)
# # starting value, and how much we want to increase by
# perplexity = 10
# perplexity_increase = 5
# fig, axes = plt.subplots(2, 5, figsize=(18, 4), squeeze=False)
# for ax, row_name in zip(axes[:, 0], ["t-SNE", "t-SNE"]):
#     ax.set_ylabel(row_name, rotation=0, size="large", labelpad=30)

# for row in [0, 1]:
#     for count in range(5):
#         print("Count", count)
#         tsne = TSNE(perplexity=perplexity, n_components=2)
#         tsne_transform = tsne.fit_transform(subset_digits)

#         ax = axes[row, count]
#         scatter = get_scatter(tsne_transform, ax, subset_labels)

#         ax.set_title(f"Perplexity: {perplexity}")
#         ax.set_xticks([])
#         ax.set_yticks([])

#         handles, _ = scatter.legend_elements(prop="colors", alpha=1.0)
#         create_legend(fig, handles, mnist=mnist)

#         perplexity += perplexity_increase
#         fig.suptitle(
#             f"Hyperparameter Analysis - {'MNIST' if mnist else 'FMNIST'} t-SNE -  Sample Count: {subset_size} Perplexity Increase={perplexity_increase}"
#         )
# # plt.show()
# plt.savefig(f"{'mnist' if mnist else 'fmnist'}_tsne.pdf")
# ===================================================================================================================


# ============================ CODE for Question 1 -- Variability for t-SNE, using + plotting init=random =============
# fig, axes = plt.subplots(1, 5, figsize=(18, 4), squeeze=False)

# for ax, row_name in zip(axes[:, 0], ["t-SNE"]):
#     ax.set_ylabel(row_name, rotation=0, size="large", labelpad=30)
# for count in range(1):
#     # Use random here instead of pca
#     tsne = TSNE(n_components=2, perplexity=30, init="random")
#     tsne_transform = tsne.fit_transform(digits)
#     ax = axes[0, count]
#     scatter = get_scatter(tsne_transform, ax, labels)

#     ax.set_title(f"Run {count+1}")
#     # TODO: need to figure out to add legend
#     ax.set_xticks([])
#     ax.set_yticks([])

#     handles, _ = scatter.legend_elements(prop="colors", alpha=1.0)
#     create_legend(fig, handles, mnist)
#     fig.suptitle(f"t-SNE on {'MNIST' if mnist else 'FMNIST'} Data Set -- init=random")
# plt.show()
