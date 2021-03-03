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



class Distribution_anq():

    def __init__(self,na,threshold):
        add_program(na)

        print("---------アンケート分類中--------")

        self.root_data = na + "/anq/"

        f = open("./Model/"+na+".pickle",mode = "rb")
        train_data = pickle.load(f)

        new_id = len(train_data)
        face_encoding = [d.get("encoding")for d in train_data]
        face_id = [d.get("id")for d in train_data]
        folder = glob2.glob("./anq_data/*")


        for t in range(len(folder)):
            self.folderName = os.path.split(folder[t])[1]
            images_test = glob2.glob("./anq_data/"+self.folderName+"/*.jpg")

            print(self.folderName,": Number of Images:",len(images_test))

            for k in tqdm(images_test):
                moon = []
                face_encoding = [d.get("encoding")for d in train_data]
                face_id = [d.get("id")for d in train_data]
                self.test_image_name = os.path.basename(k)
                self.csv_name = self.test_image_name[0:-4]+".csv"

                im = cv2.imread(k)
                im = cv2.resize(im,(int(im.shape[1]*0.5),int(im.shape[0]*0.5)))
                face_location = face_recognition.face_locations(im)
                
                if face_location == []:
                    locationSize = 0
                else:
                    locationSize = list(face_location[0])[1]-list(face_location[0])[3]
                    
                if  locationSize <=2:
                    pass
                else:
                    biden_face_encoding = face_recognition.face_encodings(im)[0]
                    for i in range(len(face_encoding)):
                        result = face_recognition.compare_faces([face_encoding[i]],biden_face_encoding)
                        distance = face_recognition.face_distance([face_encoding[i]],biden_face_encoding)
                        moon.append(distance)

                    min_encording = min(moon)

                    ida = int(moon.index(min(moon)))
                    self.min_id = face_id[ida]
                
                    if min_encording > threshold:
                        
                        self.unknown_subject_id()
                    else:

                        self.subject_id()
        
        try:
            shutil.rmtree(folder[t]+"/")
        except:
            print("./anq_dataの中にフォルダがありません")
        


        print("---------分類終了--------")


    def subject_id(self):
        if not os.path.exists("./subject/"+self.root_data+self.min_id):
            os.mkdir("./subject/"+self.root_data+self.min_id)
        try:
            shutil.move("./anq_data/"+self.folderName+"/"+self.csv_name,"./subject/"+self.root_data+self.min_id+"/"+self.min_id+"_"+self.csv_name)
            shutil.move("./anq_data/"+self.folderName+"/"+self.test_image_name,"./subject/"+self.root_data+self.min_id+"/"+self.min_id+"_"+self.test_image_name)
        except:

            if not os.path.exists("./subject/"+self.root_data+"No_csvFile_data"):
                os.mkdir("./subject/"+self.root_data+"No_csvFile_data")
            shutil.move("./anq_data/"+self.folderName+"/"+self.test_image_name,"./subject/"+self.root_data+"No_csvFile_data/"+self.test_image_name)


    def unknown_subject_id(self):
        if not os.path.exists("./subject/"+self.root_data+"Unknown"):
            os.mkdir("./subject/"+self.root_data+"Unknown")
        
        try:
            shutil.move("./anq_data/"+self.folderName+"/"+self.csv_name,"./subject/"+self.root_data+"Unknown/"+self.csv_name)
            shutil.move("./anq_data/"+self.folderName+"/"+self.test_image_name,"./subject/"+self.root_data+"Unknown/"+self.test_image_name)
        
        except:
            print(self.csv_name,"がありません")
            if not os.path.exists("./subject/"+self.root_data+"No_csvFile_data"):
                os.mkdir("./subject/"+self.root_data+"No_csvFile_data")
            shutil.move("./anq_data/"+self.folderName+"/"+self.test_image_name,"./subject/"+self.root_data+"No_csvFile_data/"+self.test_image_name)
            


 