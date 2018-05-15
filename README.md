### Install the following dependencies:

1. Install Python 3
```bash
$ brew install python3
```

2. Install dependent libraries using pip3
```bash
$ pip3 install tensorflow flask
```

3. Start the server
```bash
$ python3 api.py
```

4. To view sample predictions, visit:
```
http://127.0.0.1:5000/api/predict?features=/api/predict?features=[[0.1,4.2,0.3,1.4],[1.2,2.3,3.1,4.2],[0.1,4.2,0.3,1.4]]
```
