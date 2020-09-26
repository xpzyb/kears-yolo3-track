 # -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import logging

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416,416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'), file=open ( 'groundtruth.txt' , 'a' ))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        class1 = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            class1.append(predicted_class )


            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        class1.reverse()
        return out_boxes, out_scores, class1

    def close_session(self):
        self.sess.close()

'''
追踪程序
'''
def detect_video(yolo, video_path, output_path=""):
    import time
    vid = cv2.VideoCapture(video_path)
    Points = []
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    loop = 10
    frame = 0

    while True:
        return_value, img = vid.read()
        if not return_value:
            print ( "The video is over!" )

            return
        if frame % loop == 0:
            Dtime1 = time.time()
            image = Image.fromarray(img)
            loc, scores, lab = yolo.detect_image(image)
            result = np.asarray(image)
            Dtime2 = time.time()
            Dtime = Dtime2 - Dtime1
            info = "检测时间: %.2f ms" % (1000 * Dtime)
            print(info, file=open('time.txt', 'a'))
            initimg = result
            Points = []

        else:
            loc, scores, lab = tracking ( initimg , img , loc, scores, lab, frame,Points)
            result = np.asarray ( image )
        cv2.imshow ( "result" , img )

        if isOutput:
            out.write ( result )

        if cv2.waitKey ( 1 ) & 0xFF == ord ( 'q' ):
            break

        frame = frame + 1
        frame = frame % loop
    yolo.close_session ()
'''
追踪函数
'''
def tracking(initImg, img, loc, scores, lab, frame,Points):
    import time
    Ttime1 = time.time()
    COLORS = np.random.randint ( 0 , 255 , size=(80 , 3) , dtype='uint8' )
    if frame == 1:
        global tracker  # 这步很重要，每次初始化跟踪时需要清除原先所跟踪的目标；否则，跟踪的目标会累加

        tracker = cv2.MultiTracker_create ()
        for i , newbox in enumerate (loc ):
          ok = tracker.add(cv2.TrackerKCF_create(), initImg, (int(newbox[1]), int(newbox[0]), int (newbox[3]), int (newbox[2])))
          if not ok:
            print("The tracker initialization failed!")
            return
    ok, boxes = tracker.update(img)
    Ttime2 = time.time()
    Ttime = Ttime2 - Ttime1
    info1 = "跟踪时间: %.2f ms" % (1000 * Ttime)
    print(info1, file=open('time.txt', 'a'))
    DrawTime1 = time.time()
    if ok:

        loc = boxes
        num = 0
        for i,box in enumerate (boxes):
            temp = np.array([int ( box[0] + (box[2]-box[0]) *0.5 ) , int ( box[1] + (box[3]-box[1]) *0.5 )])
            Points.append(temp)
            bbox_color  = [int ( c ) for c in COLORS[i]]

            c1,c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(img, c1,c2, bbox_color, 2)
            text = '{}: {:.2f}'.format(lab[i], scores[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (int(box[0]), int(box[1]) - text_h - baseline), (int(box[0]) + text_w, int(box[1])), bbox_color, -1)
            cv2.putText(img, text, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print ( lab[i] ,'{:.2f}'.format(scores[i]), c1 ,c2 , file=open ( 'groundtruth.txt' , 'a' ))
            num = num + 1
            if len ( Points ) > num:
                for i in range ( len ( Points ) - num ):
                    x0 = Points[i][0]
                    y0 = Points[i][1]
                    x1 = Points[i + num][0]
                    y1 = Points[i + num][1]
                    distant = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                    if distant < 10:
                        cv2.circle(img,(x0,y0),5,bbox_color,-1)
                        cv2.line(img,(x0,y0),(x1,y1), bbox_color,4)
    DrawTime2 = time.time()
    DrawTime = DrawTime2 - DrawTime1
    info2 = "绘制轨迹时间: %.2f ms" % (1000 * DrawTime)
    print ( info2 , file=open ( 'time.txt' , 'a' ) )



    return loc, scores, lab

