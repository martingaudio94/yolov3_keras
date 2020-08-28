from keras_ocr.detection import Detector
from keras_ocr.recognition import Recognizer
from keras_ocr.tools import read

import numpy as np

from yolo_ocr_TML import yolo_ocr,model
import os
import cv2


detector=Detector()
recognizer=Recognizer()

temp=r'C:\Users\Usuario1\Desktop\CODIGO EN python\temp'
image=r'C:\Users\Usuario1\Desktop\muestra_procesadas\Imagenes'
direct=[x for x in os.scandir(r'C:\Users\Usuario1\Desktop\muestra_procesadas\Imagenes')]

imgs=yolo_ocr(model,image,direct[0].name)

keys=imgs.keys()

for i in keys:
    cv2.imwrite(temp+'\\'+str(i)+'.jpg',imgs[i])
temp_dir=os.scandir(temp)

print('probando reconocimiento')
campos={}

for i in temp_dir:
    #img=cv2.imread(i.path,cv2.IMREAD_COLOR)
    img = cv2.imread(i.path)
    boxes=detector.detect(images=[img])[0]
    predict=recognizer.recognize_from_boxes(image=img,boxes=boxes)
    letters=' '.join([x[0] for x in predict])
    campos[i.name.split('.')[0]]=letters
