import requests
import json
import cv2
import numpy as np
while True:
	r = requests.get('http://127.0.0.1:8080/get_frame',headers = {'cam_id': 'id1'})
	r2 = requests.get('http://127.0.0.1:8080/get_frame',headers = {'cam_id': 'id2'})
	
	data = r.content
	frame = json.loads(data.decode("utf8"))
	frame = np.asarray(frame, np.uint8)
	frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
	
	data2 = r2.content
	frame2 = json.loads(data2.decode("utf8"))
	frame2 = np.asarray(frame2, np.uint8)
	frame2 = cv2.imdecode(frame2, cv2.IMREAD_COLOR)

	video_feeds = np.hstack((frame, frame2))
	cv2.imshow('im', video_feeds)
	if cv2.waitKey(1) == ord('q'):
			break
cv2.destroyAllWindows()

