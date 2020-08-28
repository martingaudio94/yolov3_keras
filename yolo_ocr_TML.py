from yolo import YOLO
from PIL import Image
import numpy as np
from os import scandir
import cv2
import pytesseract
import keras_ocr
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
 
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

###comenzamos
print("Comienzo del proceso . . .")
print("Leyendo directorio")
base=r'C:\Users\Usuario1\Desktop\tml_entrenamiento\jpgs_p_caracteres'
direct=scandir(base)
imagenes=r'C:\Users\Usuario1\Desktop\tml_entrenamiento\jpgs_p_caracteres'
recortes=r'C:\Users\Usuario1\Desktop\tml_entrenamiento\caracteres'
output_ocr=r'C:\Users\Usuario1\Desktop\tml_entrenamiento\caracteres'
#for i in direct:
#    if i.is_file():
#        img=Image.open(r'{}\{}'.format(base,i.name))
#        img_arr=np.array(img)
#        imgs,coord=YOLO.detect_image(YOLO(image=True),img)
def get_string(img_path,name,output_ocr):
        img=cv2.imread(img_path)
    #    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        unos=np.ones((1,1),np.uint8)
        img=cv2.dilate(img,unos,iterations=1)
        img=cv2.erode(img,unos,iterations=1)

    #    cv2.imwrite(r'C:\Users\Usuario1\Desktop\recortes\{}_witout_noise.jpg'.format(name),img)
    #    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        #results=pytesseract.image_to_string(Image.open(r'C:\Users\Usuario1\Desktop\recortes\{}'.format(name)))
        results=pytesseract.image_to_string(img)
        with open(r'{}.txt'.format(output_ocr+'\\'+name)) as file:
                file.write(str(results))
        return 'OK'

#dirr=[x for x in direct]
print("cargando modelo")
model=YOLO(image=True)

def yolo_ocr(model,base,directorio):
        kkk=0
        images={}
        curdir=os.path.abspath(os.curdir)

        print("leyendo imagenes")
        img=Image.open(base+'\\'+directorio)
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img)
        img=background
        img_arr=np.array(img)
        imgi,coord=YOLO.detect_image(model,img)       
        imagenes={}
        ###recortando
        print("recortando")
        for i in range(len(coord['clase'])):
            cordin=coord['coor'][i]
            b=0
            r=0
            t=0
            l=0
                                
            arr=img_arr[cordin[0][0]+10+t:cordin[0][1]+b,
                        cordin[1][0]+10+l:cordin[1][1]+r]
            imagenes[coord['clase'][i][:len(coord['clase'][i])-4]]=arr
                
            kkk+=1
        return imagenes
print("Modelo cargado y preparado, ejectuando la funcionalidad...")
#yolo_ocr(model,direct)
