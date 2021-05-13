from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch


def get_loader(dataset_dir='dataset', image_size=256, batch_size=1, num_workers=0):
    transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageFolder(dataset_dir, transform=transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers)
    return data_loader