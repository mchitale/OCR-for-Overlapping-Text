import os
import cv2
import numpy as np
import csv

root = "C:\\Users\\machital\\Desktop\\OCR_Ovl\\English"

list_dir = os.listdir(root)

fnt = list_dir[0]
hnd = list_dir[1]
img = list_dir[2]

fnt_dir = os.listdir(os.path.join(root,fnt))
hnd_dir = os.listdir(os.path.join(root,hnd))
img_dir = os.listdir(os.path.join(root,img))


bmpb = os.listdir(os.path.join(root,img,img_dir[0]))
bmpg = os.listdir(os.path.join(root,img,img_dir[1]))

bad_img_dir = os.listdir(os.path.join(root,img,img_dir[0],bmpb[0]))
good_img_dir = os.listdir(os.path.join(root,img,img_dir[1],bmpg[0]))

hnd_img_dir = os.listdir(os.path.join(root,hnd,hnd_dir[1]))
del hnd_img_dir[0]

#Final directories are fnt_dir, hnd_img_dir, bad_img_dir, good_img_dir
filen = open('pixels.csv','w',newline='')
for i,folder in enumerate(fnt_dir):
    if i in range(0,10):
        label = i
    elif i in range(10,36):
        label = chr(ord('A')+(i-10))
    else:
        label = chr(ord('a')+(i-36))

    

    for i,image in enumerate(os.listdir(os.path.join(root,fnt,folder))):
       
        img = cv2.imread(os.path.join(root,fnt,folder,image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rs = cv2.resize(img, (32,32))
        flat = rs.flatten('C')
        row = []
        writer= csv.writer(filen, delimiter=',')
        for x in np.nditer(flat.T, order='C'): 
            row.append(str(x))

        filen.write(str(label))
        filen.write(str(','))       
        writer.writerow(row)


            