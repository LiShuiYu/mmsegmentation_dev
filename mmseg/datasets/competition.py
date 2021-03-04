import os

import mmcv
from mmcv.utils.logging import print_log

from .builder import DATASETS
from .custom import CustomDataset
from mmseg.utils import get_root_logger

@DATASETS.register_module()
class SuiChangDataset(CustomDataset):

    CLASSES = ['Cultivated', 'Wood', 'Grass', 'Road', 'Urban', 
               'Rural', 'Industrial', 'Building', 'Waters', 'Bare']

    PALETTE = [[255, 185, 15], [0, 100, 0], [0, 255, 0], 
               [255, 255, 255], [119, 136, 153], 
               [255, 0, 0], [255, 255, 0], [255, 181, 197], 
               [0, 0, 255], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(SuiChangDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
    
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, 
                         split):
        
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=os.path.join(img_dir, img_name + img_suffix))
                    if ann_dir is not None:
                        seg_map = os.path.join(ann_dir, img_name + seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class SuiChangRoadDataset(CustomDataset):

    CLASSES = ['Background', 'Road']

    PALETTE = [[0, 0, 0], [0, 255, 0]]

    def __init__(self, **kwargs):
        super(SuiChangRoadDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)