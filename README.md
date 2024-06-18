# drone
### This project focuses on the detection of drones using thermal imagery, leveraging computer vision techniques and deep learning models.

#### The repository includes scripts for training YOLO v3 models on thermal drone data, as well as scripts for real-time detection on video streams.

#### Installation: Make sure to install the necessary dependencies listed in requirements.txt before running the scripts. Use pip install -r requirements.txt.

Usage:

    Training: To train the YOLO v3 model, configure the training options in train.py, set up the dataset paths, and run python train.py.
    
Dataset: Ensure your dataset is structured according to YOLO's requirements, with annotated bounding boxes for drone locations in text files associated with each image.