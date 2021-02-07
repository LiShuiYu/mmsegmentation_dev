from torch.utils.data.dataset import Dataset
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class SuiChangDataset(CustomDataset):

    CLASSES = ['Cultivated', 'Wood', 'Grass', 'Road', 'Urban', 
               'Rural', 'Industrial', 'Building', 'Waters', 'Bare']

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], 
               [139, 101, 8], [54, 100, 140], [160, 82, 45], [49, 76, 32], [88, 91, 96]]

    def __init__(self, **kwargs):
        super(SuiChangDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)