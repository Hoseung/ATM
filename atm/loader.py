from PIL import Image
from typing import Any, Callable, Optional, Tuple
import torchvision
from torchvision.datasets.vision import VisionDataset
from functools import partial
import numpy as np

class TonemapImageDataset(VisionDataset):
    def __init__(self, 
                 data_array, 
                 tmo,
                 labels: Optional = None, 
                 train: bool=True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,):
        self._array = data_array
        self._good_gids = np.array([gal['img_name'] for gal in data_array])
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.tmo = tmo
        self._bad_tmo=False

    def _apply_tm(self, image):
        try:
            return self.tmo(image)
        except ZeroDivisionError:
            print("division by zero. Probably bad choice of TM parameters")
            self._bad_tmo=True
            return image

    def _to_8bit(self, image):
        """
        Normalize per image (or use global min max??)
        """

        image = (image - image.min())/image.ptp()
        image *= 255
        return image.astype('uint8')        
    
    def __len__(self) -> int:
        return len(self._array)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        For super
        """
        image, _segmap, weight = self._array[idx]['data']
        image[~_segmap.astype(bool)] = 0#np.nan # Is it OK to have nan?
        image[image < 0] = 0

        image = self._to_8bit(self._apply_tm(image))
        image = Image.fromarray(image)
        label = self.img_labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


    
class TonemapImageDatasetPair(TonemapImageDataset):
    """
    returns two differently (randomly) transformed version of an image.
    
    img_labels = np.ndarray
    """
    def __getitem__(self, idx):
        image, _segmap, weight = self._array[idx]['data']
        image[~_segmap.astype(bool)] = 0#np.nan # Is it OK to have nan?
        image[image < 0] = 0
        
        image = self._to_8bit(self._apply_tm(image))
        image = Image.fromarray(image)
        label = self.img_labels[idx]
        
        if self.transform is not None:
            im_1 = self.transform(image) # random transform. 
            im_2 = self.transform(image)

        return (im_1, im_2), label
        