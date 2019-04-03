from app import app, db
from app.models import Yaml, Frame
from flask import Flask, session
import cv2
import yaml
import numpy as np

img = cv2.imread('test.jpg')
image = cv2.resize(img, None, fx=0.6, fy=0.6)
refPt = []
data = []
cropping = False

def yaml_loader(file_path):
    with open(file_path, "a+") as file_descr:
        data = yaml.load(file_descr)
        return data

def yaml_dump(file_path, data):
    with open(file_path, "a+") as file_descr:
        yaml.dump(data, file_descr)


def click_and_crop(event, x, y, flags, param):
    
    file_path = 'lot_config.yml'

    info = {'id': 0, 'points': []}
    global refPt

    if event == cv2.EVENT_LBUTTONDBLCLK:
        refPt.append((x,y))
        cropping = False

    if len(refPt) == 4:
        if data == []:
            if yaml_loader(file_path) != None:
                new_data = len(yaml_loader(file_path))
            else:
                new_data = 0
        else:
           if yaml_loader(file_path) != None:
               new_data = len(data) + len(yaml_loader(file_path))
           else:
               new_data = len(data)

        cv2.line(image, refPt[0], refPt[1], (0, 0, 255), 2)
        cv2.line(image, refPt[1], refPt[2], (0, 0, 255), 2)
        cv2.line(image, refPt[2], refPt[3], (0, 0, 255), 2)
        cv2.line(image, refPt[3], refPt[0], (0, 0, 255), 2)

        corner_1 = list(refPt[2])
        corner_2 = list(refPt[3])
        corner_3 = list(refPt[0])
        corner_4 = list(refPt[1])

        info['points'] = [corner_1, corner_2, corner_3, corner_4]
        info['id'] = new_data + 1
        data.append(info)
        refPt = []

def main():
    file_path = 'lot_config.yml'
    print('Enter ID of CAMERA:')
    frameid = input()
    frame_query = db.session.execute("SELECT frame_fname FROM Frame WHERE id = :frame_id", {"frame_id": frameid}).fetchall()
    for row in frame_query:
        d = dict(row.items())
    print(d['frame_fname'])
    img = cv2.imread(d['frame_fname'])
    image = cv2.resize(img, None, fx=0.6, fy=0.6)
    
    cv2.namedWindow("Click to mark points")
    cv2.imshow("Click to mark points", image)
    cv2.setMouseCallback("Click to mark points", click_and_crop)

    while True:
        cv2.imshow("Click to mark points", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    # data list into yaml file
    if data != []:
        yaml_dump(file_path, data)
    cv2.destroyAllWindows()

    u = Yaml(yaml_fname=file_path)
    db.session.add(u)
    db.session.commit() 

if __name__ == '__main__':
    main()
