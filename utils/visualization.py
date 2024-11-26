import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def plot_confusion_matrix(y_true, y_pred, title, label_tags):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_tags)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


# Function to plot decision boundaries for SVM
def plot_decision_boundaries(X, y, model, title, ax):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)


def visualize(model, test_loader, label_tags, title, device):
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()
    preencoded_features = []
    encoded_features = []
    labels_list = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            preencoded = images.view(images.size(0), -1).cpu().numpy()
            preencoded_features.append(preencoded)

            encoded = feature_extractor(images).cpu().numpy()
            encoded_features.append(encoded)

            labels_list.append(labels.numpy())

            del images, encoded
            torch.cuda.empty_cache()

    encoded_features = np.concatenate(encoded_features)
    labels_list = np.concatenate(labels_list)

    tsne_2d = TSNE(n_components=2, random_state=42)
    encoded_tsne_2d = tsne_2d.fit_transform(encoded_features)
    fig, axes = plt.subplots(figsize=(10, 6))

    # Define a discrete colormap with boundaries based on unique labels
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(len(unique_labels) + 1) - 0.5, ncolors=len(unique_labels)
    )

    scatter1 = axes.scatter(
        encoded_tsne_2d[:, 0],
        encoded_tsne_2d[:, 1],
        c=labels_list,
        cmap=cmap,
        norm=norm,
        alpha=0.6,
    )
    axes.set_title(title)
    cbar1 = fig.colorbar(scatter1, ax=axes, shrink=0.7, ticks=unique_labels)
    cbar1.set_label("Labels")
    cbar1.set_ticks(unique_labels)
    cbar1.set_ticklabels(label_tags)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.tight_layout()
    plt.show()

    return feature_extractor


def plot_tsne_2d(
    preencoded,
    encoded,
    labels,
    axes,
    label_tags=["class 1", "class 2", "class 3"],
    fig=None,
):
    # Define a discrete colormap with boundaries based on unique labels
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(len(unique_labels) + 1) - 0.5, ncolors=len(unique_labels)
    )

    scatter1 = axes[0].scatter(
        preencoded[:, 0], preencoded[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.6
    )
    axes[0].set_title(f"Pretrained features - 3D")
    cbar1 = fig.colorbar(scatter1, ax=axes[0], shrink=0.7, ticks=unique_labels)
    cbar1.set_label("Labels")
    cbar1.set_ticks(unique_labels)
    cbar1.set_ticklabels(label_tags)

    scatter2 = axes[1].scatter(
        encoded[:, 0], encoded[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.6
    )
    axes[1].set_title(f"Encoded Features - 2D")
    cbar2 = fig.colorbar(scatter2, ax=axes[1], shrink=0.7, ticks=unique_labels)
    cbar2.set_label("Labels")
    cbar2.set_ticks(unique_labels)
    cbar2.set_ticklabels(label_tags)


def plot_tsne_3d(
    preencoded, encoded, labels, label_tags=["class 1", "class 2", "class 3"], fig=None
):
    # Define a discrete colormap with boundaries based on unique labels
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(len(unique_labels) + 1) - 0.5, ncolors=len(unique_labels)
    )

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    scatter1 = ax1.scatter(
        preencoded[:, 0],
        preencoded[:, 1],
        preencoded[:, 2],
        c=labels,
        cmap=cmap,
        norm=norm,
        alpha=0.6,
    )
    ax1.set_title(f"Pretrained features - 3D")
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.7, ticks=unique_labels)
    cbar1.set_label("Labels")
    cbar1.set_ticks(unique_labels)
    cbar1.set_ticklabels(label_tags)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    scatter2 = ax2.scatter(
        encoded[:, 0],
        encoded[:, 1],
        encoded[:, 2],
        c=labels,
        cmap=cmap,
        norm=norm,
        alpha=0.6,
    )
    ax2.set_title(f"Encoded Features - 2D")
    cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.7, ticks=unique_labels)
    cbar2.set_label("Labels")
    cbar2.set_ticks(unique_labels)
    cbar2.set_ticklabels(label_tags)
