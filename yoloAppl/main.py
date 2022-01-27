import sys
import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets, uic

Ui_Form = uic.loadUiType("./main.ui")[0]


class main(QtWidgets.QMainWindow, Ui_Form):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUi()

    def initUi(self):
        # self.setGeometry(800, 200, 300, 300)
        self.setWindowTitle('demo')
        self.pushButton.clicked.connect(self.fileOpen)

    def fileOpen(self):
        global filename
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')

        if ".jpg" in filename[0] or ".png" in filename[0]:
            self.loadYolo(filename[0])

    def loadYolo(self, filename):

        # Yolo 로드
        self.net = cv2.dnn.readNet("./yolo/test_best.weights", "./yolo/test.cfg")
        # print(self.net)
        self.classes = []
        with open("./yolo/obj.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 이미지 가져오기
        self.img = cv2.imread(filename)
        self.img = cv2.resize(self.img, None, fx=0.4, fy=0.4)
        self.height, self.width, self.channels = self.img.shape
        print(self.height, self.width, self.channels)

        self.detectImage()

    def detectImage(self):

        # Detecting objects
        blob = cv2.dnn.blobFromImage(self.img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # print(outs)

        # 정보를 화면에 표시
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    # 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    print(self.classes[class_id])
                    print(confidence)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[i]
                cv2.rectangle(self.img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.img, label, (x, y + 30), font, 3, color, 3)
                # print(label)
        cv2.imshow("output", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = main()
    main.show()
    app.exec_()
