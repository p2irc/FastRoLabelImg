from .coco import CocoDataset


class MyDataset(CocoDataset):

    CLASSES = ('wheat_head')
