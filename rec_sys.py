#!usr/bin/env python3
import cv2
import numpy as np
from color import color_chart_new


class Analysis():
    def __init__(self):
        pass

    def white_balence(self, img_np):
        # channel average
        b, g, r = cv2.split(img_np)
        b_avg = cv2.mean(b)[0]
        g_avg = cv2.mean(g)[0]
        r_avg = cv2.mean(r)[0]
        
        # bias
        k = (b_avg + g_avg + r_avg) / 3
        kb = k / b_avg
        kg = k / g_avg
        kr = k / r_avg
        
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        
        img_wb = cv2.merge([b, g, r])

        # resize
        scale = 0.5
        width = int(img_wb.shape[1] * scale)
        height = int(img_wb.shape[0] * scale)
        img_wb = cv2.resize(img_wb, (width, height), cv2.INTER_AREA)

        return img_wb 

    def check(self, img_np):
        # face & eyes detection
        # https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale

        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img_np[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        if len(faces) == 1 and len(eyes) == 2:
            return roi_color, eyes
        else:
            return False, False

    def anylyze(self, roi_color, eyes):
        # eyes[0] : right eye, eyes[1] : left eye
        x1, y1, w, h, x2, y2 = eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3], eyes[1][0], eyes[1][1]
        half_w = int(w / 2)
        half_h = int(h / 2)

        # check point
        ckpt_a = roi_color[y1+h:y1+h+half_h, x1:x1+half_w]  # right cheek
        ckpt_b = roi_color[y2+h:y2+h+half_h, x2+half_w:x2+w]  # left cheek
        ckpt_c = roi_color[y1-half_h:y1, x1+w:x1+w+half_w]  # forehead

        ckpt_a = cv2.cvtColor(ckpt_a, cv2.COLOR_BGR2HSV)
        ckpt_b = cv2.cvtColor(ckpt_b, cv2.COLOR_BGR2HSV)
        ckpt_c = cv2.cvtColor(ckpt_c, cv2.COLOR_BGR2HSV)

        ckpt_a = np.array(cv2.mean(ckpt_a))
        ckpt_b = np.array(cv2.mean(ckpt_b))
        ckpt_c = np.array(cv2.mean(ckpt_c))

        # color chart comparison
        res_color, res_norm = '', -1
        for key, value in color_chart_new.items():
            norm_ckpt_a = np.linalg.norm(ckpt_a - value)
            norm_ckpt_b = np.linalg.norm(ckpt_b - value)
            norm_ckpt_c = np.linalg.norm(ckpt_c - value)
            norm = (norm_ckpt_a + norm_ckpt_b + norm_ckpt_c) / 3

            if res_norm < 0 or norm < res_norm:
                res_color = key
                res_norm = norm
        
        return res_color