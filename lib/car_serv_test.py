import requests
import json
import cv2
import numpy as np
from img_proc import ImgProcessor

r = requests.get('http://127.0.0.1:8080/get_frame')
data = r.content
frame = json.loads(data.decode("utf8"))
frame = np.asarray(frame, np.uint8)
frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

car_dect = ImgProcessor()
car = car_dect.process_frame(frame)
print(car)
