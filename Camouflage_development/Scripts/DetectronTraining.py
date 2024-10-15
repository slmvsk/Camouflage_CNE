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
    # This function is used as a placeholder to return the dataset provided to it.
    return dset


# Paths and dataset settings
annotationFolder = '/work/ReiterU/olivier/DetectronTraining/development_all/'  # Directory where annotations and data are stored
segmentation_titles = ['cuttlefish']  # List of segmentation classes (in this case, 'cuttlefish')
currDataset_name = 'development_all'  # Name of the current dataset

# Loading the annotations (COCO format) for the dataset
obj_text = codecs.open(annotationFolder + 'annotations_coco.json', 'r', encoding='utf-8').read()
dataset = json.loads(obj_text)  # Load annotations from JSON format

# To correct file paths in the dataset
for i in range(len(dataset)):
    filename = dataset[i]['file_name']
    newFilename = annotationFolder + filename.split('/')[-1]  # Corrects the path to the image files by appending the right directory
    dataset[i]['file_name'] = newFilename  # Updates the file path in the dataset

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
TEST_SIZE = 0.1  # Set the test size (10% of the data)
RANDOM_SEED = np.random.randint(1000)  # Set a random seed for reproducibility
X_train, X_test = train_test_split(dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED)  # Splitting the data into train and test sets
print(len(X_train), len(X_test))  # Print the number of images in train and test sets

# Register the train and test datasets in Detectron2's DatasetCatalog
DatasetCatalog.register(currDataset_name + '_train', lambda: my_dataset_function(X_train))
DatasetCatalog.register(currDataset_name + '_test', lambda: my_dataset_function(X_test))

# Set metadata for the training dataset (e.g., object classes)
meta = MetadataCatalog.get(currDataset_name + '_train').set(thing_classes=segmentation_titles)
# Optional: You can set keypoint names and flip map for pose estimation (commented out)
# MetadataCatalog.get(currDataset_name + '_train').set(keypoint_names=['head', 'tail'])
# MetadataCatalog.get(currDataset_name + '_train').set(keypoint_flip_map=[('head', 'head'),('tail','tail')])

# Configuring the model
cfg = get_cfg()
# Load base configuration file for Mask R-CNN (pretrained on COCO dataset)
cfg.merge_from_file("/apps/unit/ReiterU/olivier/detectron2Configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = (currDataset_name + '_train',)  # Set the training dataset
cfg.DATASETS.TEST = (currDataset_name + '_test',)    # Set the test dataset (for inference, not training)
cfg.OUTPUT_DIR = annotationFolder + 'output'  # Output directory for saving model and results
cfg.DATALOADER.NUM_WORKERS = 8  # Number of parallel workers for data loading
cfg.SOLVER.IMS_PER_BATCH = 4  # Number of images per batch during training
cfg.SOLVER.BASE_LR = 0.001  # Base learning rate for the optimizer
cfg.SOLVER.WARMUP_ITERS = 1000  # Number of iterations for learning rate warmup
cfg.SOLVER.MAX_ITER = 4000  # Maximum number of iterations to run the training
cfg.SOLVER.STEPS = (1000, 2000)  # Steps at which learning rate is reduced
cfg.SOLVER.GAMMA = 0.05  # Learning rate reduction factor
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"  # Pretrained weights from Detectron2 Model Zoo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # Number of proposals to sample per image
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(segmentation_titles) + 1  # Number of object classes (+1 for background)
cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save a checkpoint every 500 iterations
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # Create the output directory if it doesn't exist

# Initialize the trainer and start training
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)  # Start training from scratch (don't resume)
trainer.train()  # Train the model

# Inference (testing) after training
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Load the trained weights (final model)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85  # Set threshold for detecting objects during inference
DatasetCatalog.register(str(uuid1()), lambda: my_dataset_function(dataset))  # Register the full dataset for inference
cfg.DATASETS.TEST = (currDataset_name + '_test', )  # Set test dataset for evaluation
predictor = DefaultPredictor(cfg)  # Initialize the predictor with the trained model

# Create folder for annotated results
os.makedirs(annotationFolder + "annotated_results", exist_ok=True)

# Prepare test image paths for inference
test_image_paths = []
for i in X_test:
    test_image_paths.append(i['file_name'])  # Collect file paths for test images

# Run inference on test images and save the annotated results
for num, imageName in enumerate(test_image_paths):
    file_path = imageName  # Get the file path of the test image
    im = cv2.imread(file_path)  # Read the image
    outputs = predictor(im)  # Run inference (prediction) on the image
    v = Visualizer(
      im[:, :, ::-1],  # Convert image from BGR to RGB for visualization
      metadata=meta,  # Pass metadata (class names)
      scale=1., 
      instance_mode=ColorMode.IMAGE  # Display mode
    )
    instances = outputs["instances"].to("cpu")  # Convert results to CPU for further processing
    v = v.draw_instance_predictions(instances)  # Draw the predicted instances on the image
    result = v.get_image()[:, :, ::-1]  # Convert result back to BGR for saving
    write_res = cv2.imwrite(annotationFolder + 'annotated_results/' + str(num) + '_result.png', result)  # Save annotated result
    write_test = cv2.imwrite(annotationFolder + 'annotated_results/' + str(num) + '_example.png', im)  # Save original test image

# Clean up CUDA memory
torch.cuda.empty_cache()  # Empty the cache to free up GPU memory
torch.cuda.reset_peak_memory_stats()  # Reset memory statistics
