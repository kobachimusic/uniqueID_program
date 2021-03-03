import cv2
import face_recognition
import glob2
import os,sys
import numpy as np
import pickle
import tkinter as tk
import shutil
import time
from tqdm import tqdm
from Program.add_program import add_program

class Distribution_thermo():

    def __init__(self,na,threshold):
        add_program(na)
        #どのデータを分けるか指定
        print("---------解析データ分類中--------")


        self.root_data = na + "/thermo/"

        #読み込むModelを洗濯
        f = open("./Model/"+na+".pickle",mode = "rb")
        train_data = pickle.load(f)


        new_id = len(train_data)
        face_encoding = [d.get("encoding")for d in train_data]
        face_id = [d.get("id")for d in train_data]

        folder = glob2.glob("./thermo_data/*")

        for t in range(len(folder)):
            self.folderName = os.path.split(folder[t])[1]
            images_test = glob2.glob("./thermo_data/"+self.folderName+"/*.bmp")
            print(self.folderName,": Number of Images:",len(images_test))

            for k in tqdm(images_test):
                moon = []
                face_encoding = [d.get("encoding")for d in train_data]
                face_id = [d.get("id")for d in train_data]
                self.test_image_name = os.path.basename(k)
                self.tif_name = self.test_image_name[0:-7]+"ir.tif"
                self.txt_name = self.test_image_name[0:-7]+"meta.txt"

                biden_picture = face_recognition.load_image_file("./thermo_data/"+self.folderName+"/"+self.test_image_name)

                face_location = face_recognition.face_locations(biden_picture)
                
                if face_location == []:
                    locationSize = 0
                    
                else:
                    locationSize = list(face_location[0])[1]-list(face_location[0])[3]

                if  locationSize <=250:
                    pass


                else:

                    biden_face_encoding = face_recognition.face_encodings(biden_picture)[0]

                    for i in range(len(face_encoding)):
                        result = face_recognition.compare_faces([face_encoding[i]],biden_face_encoding)
                
                        distance = face_recognition.face_distance([face_encoding[i]],biden_face_encoding)
                        moon.append(distance)
                        
                    min_encording = min(moon)
                    #min_id = str("{0:05d}".format(moon.index(min(moon))))
                    ida = int(moon.index(min(moon)))
                    self.min_id = face_id[ida]

                    
                    image = cv2.imread(k)
                    testimage = cv2.resize(image,(int(image.shape[1]*0.4),int(image.shape[0]*0.4)))

                    #顔認識の閾値　値を下げるほど条件が厳しくなるので誤認識が減るけど，スルーする可能性も高くなる
                    if min_encording > threshold:
                        
                        self.unknown_subject_id()
                    else:

                        self.subject_id()
            shutil.rmtree(folder[t])
                    
        print("---------解析データ分類終了--------")

    def subject_id(self):
        if not os.path.exists("./subject/"+self.root_data+self.min_id):
            os.mkdir("./subject/"+self.root_data+self.min_id)

        try:
            shutil.move("./thermo_data/"+self.folderName+"/"+self.tif_name,"./subject/"+self.root_data+self.min_id+"/"+self.min_id+"_"+self.tif_name)
            shutil.move("./thermo_data/"+self.folderName+"/"+self.test_image_name,"./subject/"+self.root_data+self.min_id+"/"+self.min_id+"_"+self.test_image_name)
            shutil.move("./thermo_data/"+self.folderName+"/"+self.txt_name,"./subject/"+self.root_data+self.min_id+"/"+self.min_id+"_"+self.txt_name)
        
        except:
            if not os.path.exists("./subject/"+self.root_data+"No_tiffFile_data"):
                os.mkdir("./subject/"+self.root_data+"No_tiffFile_data")
            shutil.move("./thermo_data/"+self.folderName+"/"+self.test_image_name,"./subject/"+self.root_data+"No_tiffFile_data/"+self.test_image_name)
        
    def unknown_subject_id(self):
        if not os.path.exists("./subject/"+self.root_data+"Unknown"):
            os.mkdir("./subject/"+self.root_data+"Unknown")
        
        try:
            shutil.move("./thermo_data/"+self.folderName+"/"+self.tif_name,"./subject/"+self.root_data+"Unknown/"+self.tif_name)
            shutil.move("./thermo_data/"+self.folderName+"/"+self.test_image_name,"./subject/"+self.root_data+"Unknown/"+self.test_image_name)
            shutil.move("./thermo_data/"+self.folderName+"/"+self.txt_name,"./subject/"+self.root_data+"Unknown/"+self.txt_name)
        
        except:
            if not os.path.exists("./subject/"+self.root_data+"No_tiffFile_data"):
                os.mkdir("./subject/"+self.root_data+"No_tiffFile_data")
            shutil.move("./thermo_data/"+self.folderName+"/"+self.test_image_name,"./subject/"+self.root_data+"No_tiffFile_data/"+self.test_image_name)
            
