from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import cv2
import numpy as np
import re
import time

from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication

from utils import file_processing,image_processing
import face_recognition
import threading
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
#选择哪一块gpu,如果是-1，就是调用cpu
config = tf.ConfigProto()#对session进行参数配置
config.allow_soft_placement=True
# 如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction=0.7
#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True
#按需分配显存，这个比较重要
session = tf.Session(config=config)

resize_width = 160
resize_height = 160

class face():
    def video(self):
        cam = cv2.VideoCapture("1.mp4")
        cam.set(3, 640)
        cam.set(4, 480)
        count = 0
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        while (cam.isOpened()):
            ret, frame = cam.read()
            if ret == True:
                count += 1
                if not os.path.isdir("dataset/test_images"):
                    os.mkdir("dataset/test_images")
                if count==20:
                    cv2.imwrite("dataset/test_images/1.jpg",frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                for (x, y, w, h) in faces:
                    cv2.line(frame, (x, y + int((h) / 4)), (x, y), (0, 0, 255), 2)
                    cv2.line(frame, (x, y), (int(x + (w) / 4), y), (0, 0, 255), 2)
                    cv2.line(frame, (x + int(3 * (w) / 4), y), (w + x, y), (0, 0, 255), 2)
                    cv2.line(frame, (w + x, y), (w + x, y + int((h) / 4)), (0, 0, 255), 2)
                    cv2.line(frame, (w + x, int(y + 3 * (h) / 4)), (w + x, h + y), (0, 0, 255), 2)
                    cv2.line(frame, (w + x, h + y), (x + int(3 * (w) / 4), h + y), (0, 0, 255), 2)
                    cv2.line(frame, (int(x + (w) / 4), h + y), (x, h + y), (0, 0, 255), 2)
                    cv2.line(frame, (x, h + y), (x, int(y + 3 * (h) / 4)), (0, 0, 255), 2)
                cv2.imshow('facesFound', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cam.release()
        cv2.destroyAllWindows()
    def face_recognition_image(self,model_path,dataset_path, filename,image_path):
        # 加载数据库的数据
        time.sleep(1)
        T1=time.time()
        dataset_emb,names_list=self.load_dataset(dataset_path, filename)
        # 初始化mtcnn人脸检测
        face_detect=face_recognition.Facedetection()
        # 初始化facenet
        face_net=face_recognition.facenetEmbedding(model_path)

        image = image_processing.read_image_gbk(image_path)
        # 获取 判断标识 bounding_box crop_image
        bboxes, landmarks = face_detect.detect_face(image)
        bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height")
        if bboxes == [] or landmarks == []:
            print("-----no face")
            exit(0)
        print("-----image have {} faces".format(len(bboxes)))
        face_images = image_processing.get_bboxes_image(image, bboxes, resize_height, resize_width)
        face_images = image_processing.get_prewhiten_images(face_images)
        pred_emb=face_net.get_embedding(face_images)
        pred_name,pred_score=self.compare_embadding(pred_emb, dataset_emb, names_list)
        # 在图像上绘制人脸边框和识别的结果
        show_info=[ n+':'+str(s)[:5] for n,s in zip(pred_name,pred_score)]
        T2 = time.time()
        print((T2 - T1))
        os.remove("dataset/test_images/1.jpg")

        # self.s=UsingTest1()

        # show = cv2.resize(image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        # show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        # showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
        #                          QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        # self.s.label_3.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
        # # self.s.label_3.setPixmap(image)
        # app = QApplication(sys.argv)
        # self.s.show()

        # image_processing.show_image_bboxes_text("face_recognition", image, bboxes, show_info)
        # sys.exit(app.exec_())
        # quit()

    def load_dataset(self,dataset_path,filename):
        '''
        加载人脸数据库
        :param dataset_path: embedding.npy文件（faceEmbedding.npy）
        :param filename: labels文件路径路径（name.txt）
        :return:
        '''
        embeddings=np.load(dataset_path)
        names_list=file_processing.read_data(filename,split=None,convertNum=False)
        return embeddings,names_list

    def compare_embadding(self,pred_emb, dataset_emb, names_list,threshold=0.65):
        # 为bounding_box 匹配标签
        pred_num = len(pred_emb)
        dataset_num = len(dataset_emb)
        pred_name = []
        pred_score=[]
        for i in range(pred_num):
            dist_list = []
            for j in range(dataset_num):
                dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))
                dist_list.append(dist)
            min_value = min(dist_list)
            pred_score.append(min_value)
            if (min_value > threshold):
                pred_name.append('unknow')
                print("非安防人员，禁止闯入!")
            else:
                pred_name.append(names_list[dist_list.index(min_value)])
                b=[]
                a=open("dataall.txt","r",encoding="utf-8")
                a=a.read()
                for i in a.split("\n"):
                    b.append(i)
                for i in b:
                    for j in re.findall("(.*?):", i):
                        if pred_name[0] == j:
                            x=re.findall(":(.*)", i)[0]
                        else:
                            break
                print("{}是安防系统的安防人员，请进入！".format(x))
        return pred_name,pred_score

    def main(self):
        model_path = '20180402-114759'
        dataset_path = 'dataset/emb/faceEmbedding.npy'
        filename = 'dataset/emb/name.txt'
        image_path = 'dataset/test_images/1.jpg'
        # face_recognition_image(model_path, dataset_path, filename,image_path)
        aa = threading.Thread(target=self.video)
        bb = threading.Thread(target=self.face_recognition_image, args=(model_path, dataset_path, filename, image_path,))
        aa.start()
        bb.start()

if __name__=='__main__':
    face().main()
