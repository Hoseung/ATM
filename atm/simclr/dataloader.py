from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
#from exceptions.exceptions import InvalidDatasetSelection
from ..simclr.utils import GaussianBlur

# view generator
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod 
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, dataset_name = 'cifar10', n_views=2):
        if dataset_name == 'cifar10':
            dataset_fn = lambda: datasets.CIFAR10(self.root_folder, 
                                              train=True,
                                              transform=ContrastiveLearningViewGenerator(
                                                  self.get_simclr_pipeline_transform(32), # image size = 32
                                                  n_views),
                                              download=True)
        else:
            raise NotImplementedError('Only cifar10 dataset is supported')

        #try:
        #    dataset_fn = valid_datasets[name]
        #except KeyError:
        #    raise NotImplementedError #InvalidDatasetSelection()
        #else:
        return dataset_fn()