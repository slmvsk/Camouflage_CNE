from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import cv2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
import codecs, json 
from uuid import uuid1
import matplotlib.pyplot as plt
import numpy as np
from detectron2.data.datasets import register_coco_instances
import torch


import logging


def my_dataset_function(dset):
    return dset


annotationFolder='/work/ReiterU/olivier/DetectronTraining/development_all/'
segmentation_titles=['cuttlefish']
currDataset_name='development_all'


obj_text = codecs.open(annotationFolder + 'annotations_coco.json', 'r', encoding='utf-8').read()
dataset = json.loads(obj_text)


# to correct for new path
for i in range(len(dataset)):
    filename = dataset[i]['file_name']
    newFilename = annotationFolder + filename.split('/')[-1]
    dataset[i]['file_name'] = newFilename

#train test split
from sklearn.model_selection import train_test_split
TEST_SIZE = 0.1
RANDOM_SEED = np.random.randint(1000)
X_train, X_test= train_test_split(dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED)
print(len(X_train), len(X_test))
DatasetCatalog.register(currDataset_name + '_train', lambda: my_dataset_function(X_train))
DatasetCatalog.register(currDataset_name + '_test', lambda: my_dataset_function(X_test))


meta=MetadataCatalog.get(currDataset_name + '_train').set(thing_classes=segmentation_titles)
#MetadataCatalog.get(currDataset_name + '_train').set(keypoint_names=['head', 'tail'])
#MetadataCatalog.get(currDataset_name + '_train').set(keypoint_flip_map=[('head', 'head'),('tail','tail')])



# # #train the model
cfg = get_cfg()
cfg.merge_from_file("/apps/unit/ReiterU/olivier/detectron2Configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = (currDataset_name + '_train',)
cfg.DATASETS.TEST = (currDataset_name + '_test',)   # no metrics implemented for this dataset
cfg.OUTPUT_DIR = annotationFolder + 'output'
cfg.DATALOADER.NUM_WORKERS = 8
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 4000
cfg.SOLVER.STEPS = (1000, 2000)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"  # initialize from model zoo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(segmentation_titles)+1
cfg.SOLVER.CHECKPOINT_PERIOD = 500
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()




cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85   # set the testing threshold for this model
DatasetCatalog.register(str(uuid1()), lambda: my_dataset_function(dataset))
cfg.DATASETS.TEST = (currDataset_name + '_test', )
predictor = DefaultPredictor(cfg)

os.makedirs(annotationFolder + "annotated_results", exist_ok=True)
test_image_paths=[]
for i in X_test:
    test_image_paths.append(i['file_name'])

for num,imageName in enumerate(test_image_paths):
    file_path = imageName
    im = cv2.imread(file_path)
    outputs = predictor(im)
    v = Visualizer(
      im[:, :, ::-1],
      metadata=meta, 
      scale=1., 
      instance_mode=ColorMode.IMAGE
    )
    instances = outputs["instances"].to("cpu")
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]
    write_res = cv2.imwrite(annotationFolder + 'annotated_results/' + str(num) + '_result.png', result)
    write_test = cv2.imwrite(annotationFolder + 'annotated_results/' + str(num) + '_example.png', im)    


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
