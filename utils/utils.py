import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.visualization import (
    plot_confusion_matrix,
    plot_decision_boundaries,
)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def save_features(full_loader, feature_extractor, device):
    features = []
    labels = []

    with torch.no_grad():
        for images, target in full_loader:
            images = images.to(device)
            output = feature_extractor(images)
            features.append(output.cpu().numpy())
            labels.append(target.numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    return features, labels


def encoding(model, test_loader, device):
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

    # Concatenate features and labels
    preencoded_features = np.concatenate(preencoded_features)
    encoded_features = np.concatenate(encoded_features)
    labels_list = np.concatenate(labels_list)

    tsne_2d = TSNE(n_components=2, random_state=42)
    tsne_3d = TSNE(n_components=3, random_state=42)

    encoded_tsne_2d = tsne_2d.fit_transform(encoded_features)
    encoded_tsne_3d = tsne_3d.fit_transform(encoded_features)

    preencoded_tsne_2d = tsne_2d.fit_transform(preencoded_features)
    preencoded_tsne_3d = tsne_3d.fit_transform(preencoded_features)

    return (
        encoded_tsne_2d,
        encoded_tsne_3d,
        preencoded_tsne_2d,
        preencoded_tsne_3d,
        preencoded_features,
        encoded_features,
        labels_list,
    )


def svc_classification(
    dataset,
    feature_extractor,
    class_labels,
    model_label,
    dataset_label,
    test_size,
    seed,
    device,
):
    features, labels = save_features(dataset, feature_extractor, device)

    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_2d, labels, test_size=test_size, random_state=seed
    )

    # Training
    svm_clf = SVC(kernel="linear")
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy - {dataset_label} ({model_label}): {accuracy * 100:.2f}%")

    print(classification_report(y_test, y_pred, target_names=class_labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_decision_boundaries(
        X_train,
        y_train,
        svm_clf,
        f"SVM Classification Training Set - {dataset_label} ({model_label})",
        axes[0],
    )

    plot_decision_boundaries(
        X_test,
        y_test,
        svm_clf,
        f"SVM Classification Test Set - {dataset_label} ({model_label}",
        axes[1],
    )

    plt.tight_layout()

    plot_confusion_matrix(y_test, y_pred, f"{dataset_label}", class_labels)
