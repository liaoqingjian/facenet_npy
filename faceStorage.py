import cv2
import os
def main():
    cam = cv2.VideoCapture("1.mp4")
    cam.set(3, 640)
    cam.set(4, 480)
    count = 0
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    while (cam.isOpened()):
        ret, frame = cam.read()
        if ret == True:
            # cv2.imshow('image', frame)
            count += 1
            if count==30:
                id = input('输入拼音名字：')
                if not os.path.isdir("dataset/images/{}".format(id)):
                    os.mkdir("dataset/images/{}".format(id))

                else:
                    print("已经录入此安防人员信息！")
                    quit()
                j = "dataset/images/{}".format(id)
                cv2.imwrite(j + "/{}.jpg".format(id), frame)
                print("保存成功！")
                message=input("请输入中文名字：")
                a=open("./dataall.txt","a",encoding="utf-8")
                a.write("\n")
                a.write(id)
                a.write(":")
                a.write(message)
                a.close()
                print("信息录入成功！")
                quit()
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
            cv2.imshow('image', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
             break
    cam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
