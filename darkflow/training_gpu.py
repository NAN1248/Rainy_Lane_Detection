import numpy as np
import cv2
from darkflow.net.build import TFNet

options = {"model": "cfg/yolo_darkcars.cfg",
           "load": "bin/yolo.weights",
           "batch": 8,
           "epoch": 50,
           "gpu": 1.0,
           "train": True,
           "annotation": "ExDark_Custom_Anno/Car/",
           "dataset": "ExDark/Car/",
           "labels": "labels.txt",}

tfnet = TFNet(options)

tfnet.train()
