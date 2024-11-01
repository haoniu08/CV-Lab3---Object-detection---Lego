# CV-Lab3---Object-detection---Lego

## Object detection using Faster R-CNN (ResNet-50) 

### Faster R-CNN (ResNet-50)
To run the Faster R-CNN (ResNet-50) model, please navigate to the Faster R-CNN folder and run the following command:
```
python3 main.py
```
This folder contains the natively trained Faster R-CNN (ResNet-50) model on the Lego dataset, with a fairly small data size (as of 31 Oct 2024), and equally small epoch, batch size, and learning rate. <br>

All imgs and annotations were preprocessed and stored in the lego_dataset folder, re-labelled only in one "lego" class. <br>

As for now, the result, mAP, is painful. <br>


## The YOLOV5 and Pytorch Approach

# data_preprocessing.py
Use OpenCV for data loading and preprocessing

# label_generator.py
To use yolov5 to train the model, the annotations should be converted from xml format to txt format, and the labels for each dataset are created under the label folder respectively.

# Command to train the model
```
python train.py --data lego.yaml --weights yolov5s.pt --epochs 50 --img-size 640 --batch-size 16
```
**Run the command under the yolov5 folder**

Results of training the model are saved to /yolov5/runs/train/exp3

# To view the mAP results after training
```
python val.py --data lego.yaml --weights runs/train/exp/weights/best.pt --img-size 640 --iou-thres 0.5
```

# 	Run the following command to see how well the model performs on new images:
```
python detect.py --weights runs/train/exp/weights/best.pt --source /path/to/test/images --img-size 640 --conf-thres 0.5
```
