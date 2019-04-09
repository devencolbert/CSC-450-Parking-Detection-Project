# CSC-450-Parking-Detection-Project
This is a project that uses edge detection to detection whether a parking space is available or not. The results are displayed on a web-hosted GUI.

# Installation
1. Run the following commands in CMD

```
py -3 -m venv venv
venv\Scripts\activate
pip install -e .

set FLASK_APP=hello.py
python -m flask run
```

```
python camera_client.py
python datasets.py
python image_processing_software.py
```

To run test with Deven's files:
1. Move contents of 'test file' to root
2. Run
```
python datasets.py
python image_processing_software.py
```

# Dependencies:
* Flask
* Flask-SQLAlchemy
* Flask-Migrate
* Flask-Login
* MatPlotLib
* PyYaml
* Opencv-Python
