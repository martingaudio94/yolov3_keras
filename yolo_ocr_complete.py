from yolo import YOLO
from PIL import Image
import numpy as np
from os import scandir
import cv2
import pytesseract
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
 
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

###comenzamos
print("Comienzo del proceso . . .")
print("Leyendo directorio")
base=r'C:\Users\Usuario1\Desktop\recortes'
direct=scandir(base)
imagenes=r'C:\Users\Usuario1\Desktop\recortes'
recortes=r'C:\Users\Usuario1\Desktop\recortes'
output_ocr=r'C:\Users\Usuario1\Desktop\txts'
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
        
def yolo_ocr(model,directorio):
        kkk=0
        

        for kl in directorio:
            print("leyendo imagenes")
            img=Image.open(base+'\\'+str(kl.name))
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img)
            img=background
            img_arr=np.array(img)
            imgi,coord=YOLO.detect_image(model,img)


            print("recortando")
            for i in range(len(coord['clase'])):
                    print(coord['clase'][i])
                    with open(r'C:\Users\Usuario1\Desktop\txts\{}'.format(kl.name),'a') as file:
                            file.write(str(coord['clase'][i]))
                
        
                #imarr=Image.fromarray(arr)
                #path_to_recortes=r'{}.jpg'.format(recortes+'\\'+coord['clase'][i]+str(kkk))
                #imarr.save(path_to_recortes)
                    
                #try:
                        
                        #get_string(path_to_recortes,str(coord['clase'][i]+str(kkk)))
                #except:
                        #print('Error al utilizar sistema de ocr')
                #imagenes.append(imarr)
print("Modelo cargado y preparado, ejectuando la funcionalidad...")
yolo_ocr(model,direct)
