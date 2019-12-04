# coding = utf-8
from torch.utils.data import DataLoader
from torchvision import datasets,transforms


class Mnist(object):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])
     ]
    )
    def __init__(self, batch_size=128):
        self.batch_size = batch_size

    def train(self):
        train_data = datasets.MNIST(root="/home/ubuntu/LRP/FR-Loss-on-Mnist-master/MNIST",
                                    train=True,
                                    transform=Mnist.transform,
                                    download=False)
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_dataloader

    def test(self):
        test_data = datasets.MNIST(root="/home/ubuntu/LRP/FR-Loss-on-Mnist-master/MNIST",
                                    train=False,
                                    transform=Mnist.transform,
                                    download=False)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return test_dataloader
