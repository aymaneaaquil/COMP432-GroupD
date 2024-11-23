import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler


def data_loader(
    data_path,
    batch_size,
    random_seed=42,
    validation_size=0.1,
    test_size=0.3,
    shuffle=True,
):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )

    # define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.ImageFolder(data_path, transform=transform)

    train_data, test_data = random_split(dataset, [1 - test_size, test_size])

    # load the dataset
    training_dataset = train_data
    validation_dataset = train_data

    num_train = len(training_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    training_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, sampler=train_sampler
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return (training_loader, validation_loader, test_loader)


def full_data_loader(data_path, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )

    # define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.ImageFolder(data_path, transform=transform)

    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return full_loader
