from .coco import CocoDataset

class ActiveLearningDataset(CocoDataset):
    CLASSES = ('label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8', 'label_9')
