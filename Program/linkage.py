import os ,sys ,cv2
import glob2
import dlib
import cv2
import imutils
from imutils import face_utils
import numpy as np
import math
import shutil
import time
from tqdm import tqdm
from Program.add_program import add_program

class linkage():

    def __init__(self,na,minute):
        add_program(na)

        print("-----------紐付け中-----------")


        predictor_path = './Model/faceLandmark.dat'
        self.na = na
        self.minute = minute
        self.detector = dlib.get_frontal_face_detector() #顔検出器の呼び出し。ただ顔だけを検出する。
        self.predictor = dlib.shape_predictor(predictor_path) #顔から目鼻などランドマークを出力する
        self.anal_data = self.na


        #先頭が0から始まるフォルダーのみを取得
        anq_folder = glob2.glob('./subject/'+self.anal_data+'/anq/*')


        for folder in tqdm(anq_folder):
            label_ID = os.path.basename(folder)

            try:
                folder_figurecheck = int(label_ID) #フォルダが数字意外のときははじく
                img_file = glob2.glob(folder+'/*.jpg')
                self.create_folder(label_ID)

                thermo_floder = glob2.glob('./subject/'+self.anal_data+'/thermo/'+label_ID+'/*.bmp')

                for img in img_file:
                    file_name = os.path.basename(img)
                    date, base_time = self.calcurate_time(file_name)
                    min_time = base_time - self.minute
                    max_time = base_time + self.minute

                    count_anal_file = []


                    for bmp_img in thermo_floder:

                        bmp_img_name = os.path.basename(bmp_img)

                        date_thermo,time_thermo = self.calcurate_time(bmp_img_name)
                        
                        if date == date_thermo and min_time <= time_thermo and time_thermo < max_time :
                            count_anal_file.append(bmp_img)
                        else:
                            pass
                    
                    count = len(count_anal_file)
                    face_RateBox = [0 for k in range(count)]

                    if face_RateBox == []:
                        pass
                    else:
                    
                        for p in range(count):
                            face_RateBox[p] = self.face_orient_rate(count_anal_file[p])
                        face_RateBox  = np.array(face_RateBox)
                        face_number = face_RateBox.argmin()
                        champion_filename = os.path.splitext(count_anal_file[face_number])[0]
                        basefilename = os.path.splitext(img)[0]   
                        
                        self.move_file(champion_filename,basefilename,label_ID)
            except:
                pass
        print("-----------紐付け終了-----------")

    def calcurate_time(self,file_name):

        year = file_name[6:10]
        month = file_name[11:13]
        day = file_name[14:16]
        hour = int(file_name[17:19])
        minute = int(file_name[20:22])
        second = int(file_name[23:25])

        time = hour*3600 + minute*60 + second
        date = year + month + day

        return date,time

    def face_orient_rate(self,name):
        frame = cv2.imread(name)
        frame = imutils.resize(frame, width=1000) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        rects = self.detector(gray, 0) 
        image_points = None

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape: #顔全体の68箇所のランドマークをプロット
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

            image_points = np.array([
                    tuple(shape[30]),tuple(shape[21]),tuple(shape[22]),tuple(shape[39]),
                    tuple(shape[42]),tuple(shape[31]),tuple(shape[35]),tuple(shape[48]),
                    tuple(shape[54]),tuple(shape[57]),tuple(shape[8]),],dtype='double')

        if len(rects) > 0:
            model_points = np.array([
                    (0.0,0.0,0.0), (-30.0,-125.0,-30.0), (30.0,-125.0,-30.0), 
                    (-60.0,-70.0,-60.0), (60.0,-70.0,-60.0),(-40.0,40.0,-50.0), 
                    (40.0,40.0,-50.0),(-70.0,130.0,-100.0),(70.0,130.0,-100.0), 
                    (0.0,158.0,-10.0), (0.0,250.0,-50.0)])

            size = frame.shape

            focal_length = size[1]
            center = (size[1] // 2, size[0] // 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype='double')

            dist_coeffs = np.zeros((4, 1))

            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
            mat = np.hstack((rotation_matrix, translation_vector))

            (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
            yaw = eulerAngles[1]
            pitch = eulerAngles[0]
            roll = eulerAngles[2]

            rate = abs(yaw) + abs(pitch) + abs(roll)

            return rate

    def create_folder(self,ID):
        if not os.path.exists('./subject/'+self.anal_data+'/analysis_data/'+ID):
            os.mkdir('./subject/'+self.anal_data+'/analysis_data/'+ID)


    def move_file(self,anal_file_basename,basefilename,ID):

        filename = os.path.basename(anal_file_basename)[0:-4]
        filename_base = os.path.basename(basefilename)

        saveLocation = './subject/'+self.anal_data+'/analysis_data/'+ID+'/'

        shutil.move(anal_file_basename[0:-4]+'-rgb.bmp',  saveLocation+filename+'-rgb.bmp')
        shutil.move(anal_file_basename[0:-4]+'-ir.tif',  saveLocation+filename+'-ir.tif')
        shutil.move(anal_file_basename[0:-4]+'-meta.txt',  saveLocation+filename+'-meta.txt')

        shutil.move(basefilename+'.csv',  saveLocation+filename_base+'.csv')
        shutil.move(basefilename+'.jpg',  saveLocation+filename_base+'.jpg')

