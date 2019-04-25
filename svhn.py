import torch

import torchvision
from torch.utils.data import dataset
from torchvision.transforms import transforms

image_transform = transforms.Compose([
    transforms.ToTensor(),
])


def get_data_loader(dataset_location, batch_size):
    train_valid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    train_size = int(len(train_valid) * 0.9)
    train_set, valid_set = dataset.random_split(
        train_valid,
        [train_size, len(train_valid) - train_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return train_loader, valid_loader, test_loader
