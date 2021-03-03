import cv2
import face_recognition
import glob2
import os,sys
import numpy as np
import pickle
import time
from tqdm import tqdm


#学習するフォルダを指定


def face_Training(na,ID):
    print("--------"+na+"データの学習開始--------")
    
    data_path = "./traindata/"+na+"/"
    images = glob2.glob(data_path+"*")
    list_face_encoding = []
    k = 0
    for i ,name in tqdm(enumerate(images)):
        image_name = os.path.basename(name)

        im = cv2.imread(data_path+image_name)
        im = cv2.resize(im,(int(im.shape[1]*0.4),int(im.shape[0]*0.4)))

        #picture_of_people = face_recognition.load_image_file(data_path+image_name)
        people_face_encoding = face_recognition.face_encodings(im)
        
        if people_face_encoding == []:
            print(str(image_name))
            print("顔検出に失敗したデータです")
            print("他のデータを用いてください")
            pass
        else:
            people_face_encoding = face_recognition.face_encodings(im)[0]

            #number_id = '{0:05d}'.format(k)

            #ここで先頭5桁の数字を辞書配列に格納する
            number_id = image_name[0:ID]

            list_face_encoding.append({"id":number_id,"encoding":people_face_encoding})
            k += 1
        
    with open("./Model/"+na+".pickle",mode = "wb") as web:
        pickle.dump(list_face_encoding,web)

    print("--------"+na+"データの学習終了--------")

    return 0