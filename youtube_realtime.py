from sys import prefix
from unittest import result
import cv2, pafy
from torch import device, preserve_format 
import torch
import numpy as np
import cv2
import pafy


from time import time

LINK = "https://www.youtube.com/watch?v=UwY4q-5pRuQ&t=24s"

#doc = youtube.getbest(preftype="webm")

#video = cv2.VideoCapture(youtube.link)

#import torch  
#model = torch.load('/Users/mariochiaparini/Desktop/kaggles_machine_learning/computervision/libras.pt' )

#capture = cv2.VideoCapture(doc.link)

#check, frame = capture.read()
#print(check,frame)

#cv2.imshow("frame: ", frame)
#cv2.waitKey(10)

#capture.realease()
#cv2.destroyAllWindows()

def labels_model(self,frame, model):
    frame = [torch(frame)]
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model.to(device)
    result = self.model(frame)
    labels = result.xyxyn[0][:, -1].numpy()


class ObjectDetection:

    def __init__(self, url, out_file="Labeled_Video.avi"):
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_from_url(self):
        play = pafy.new(self._URL)
        assert play is not None
        return cv2.VideoCapture(play.url)

    def load_model(self):
        model = torch.load('/Users/mariochiaparini/Desktop/kaggles_machine_learning/computervision/libras.pt')
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        player = self.get_video_from_url()

        if (player.isOpened() == False):
            print("Error opening video")

        while (player.isOpened()):
            ret, frame = player.read()
            if ret == True:
                start_time = time()
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 3)
                format_fps = "{:.2f}".format(fps)
                print(f"Frames Per Second : {format_fps}")
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f'{format_fps} Frames per Sec', (0, int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))-100), font, 2, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        player.release()
        cv2.destroyAllWindows()


a = ObjectDetection(LINK)
a()