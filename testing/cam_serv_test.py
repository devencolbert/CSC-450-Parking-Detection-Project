import requests
import json
import cv2
import numpy as np
while True:
    r = requests.get('http://127.0.0.1:8080/get_frame')
    print(r.content)
    data = r.content
    response = requests.post('http://127.0.0.1:8090/test_config', data=data)
    frame = json.loads(data.decode("utf8"))
    frame = np.asarray(frame, np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    print(frame)
    while True:
        cv2.imshow('im', frame)
        if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()
    ans = input('Want to quit? Type q: ')
    if ans =='q':
        break
