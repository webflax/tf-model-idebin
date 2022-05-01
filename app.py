# Load Liblary
from tensorflow.keras import models
import numpy as np
import json
from flask import Flask, request, json

# Model weights are uploaded separately to the server
model_team = models.load_model('best_model_team.h5') # load model neural network team
model_risk = models.load_model('best_model_risk.h5') # load model neural network risk

def perprocess_data_team(B): # perprocess data input neural network team
    x1 ,x2, x3, x4, x5, x6 = B['B1'], B['B2'],  B['B3'], B['B4'],  B['B5'],  B['B6']
    x1 , x2 , x3 , x4 , x5 , x6 = int(x1) , int(x2)-1 , int(x3)-1 , int(x4)-1 , int(x5)-1 , int(x6)-1
    X2 , X3 , X4 , X5 , X6 = np.zeros(3) ,np.zeros(3) , np.zeros(2) , np.zeros(4) , np.zeros(2)
    X2[x2] , X3[x3] , X4[x4] , X5[x5] , X6[x6] = 1 , 1, 1, 1, 1
    X1 = x1/160
    X = np.hstack([X1 , X2 , X3 , X4 , X5 , X6])
    X = X.reshape(1,-1)
    return X

def perprocess_data_risk(C): # perprocess data input neural network team
    x1 ,x2, x3, x4, x5, x6, x7, x8 = C['C1'], C['C2'], C['C3'], C['C4'], C['C5'], C['C6'], C['C7'], C['C8']
    x1 , x2 , x3 , x4 , x5 , x6, x7, x8 = int(x1)-1 , int(x2)-1 , int(x3)-1 , int(x4)-1 , int(x5)-1 , int(x6)-1, int(x7)-1, int(x8)-1
    X1 ,X2 , X3 , X4 , X5 , X6, X7, X8 = np.zeros(4) ,np.zeros(4) , np.zeros(3) , np.zeros(3) , np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(5)
    X1[x1] ,X2[x2] , X3[x3] , X4[x4] , X5[x5] , X6[x6], X7[x7], X8[x8] = 1, 1, 1, 1, 1, 1, 1 ,1
    X = np.hstack([X1 , X2 , X3 , X4 , X5 , X6, X7, X8])
    X = X.reshape(1,-1)
    return X

app = Flask(__name__)

@app.route('/nnteam/', methods=['GET', 'POST']) #neural network team
def nnteam():
    headers = request.headers.get("API_TOKEN")
    if headers=='92fb570bf390c99e7edaf171a9692efd99abc226b0dea65484aab853663b218c':
        X = json.loads(request.data)#request.json   # get json
        X = X['data']
        per__ = {}
        for i in X:
            per__[str(i)] = np.argmax(model_team.predict(perprocess_data_team(X[str(i)])) )+1  #predict data
        return {"risk team":str(per__)}
    else:
        return "bad request"

@app.route('/nnrisk/', methods=['GET', 'POST']) #neural network risk
def nnrisk():
    headers = request.headers.get("API_TOKEN")
    if headers=='92fb570bf390c99e7edaf171a9692efd99abc226b0dea65484aab853663b218c':
        X = json.loads(request.data)#request.json   # get json
        X = X['data']
        predict_risk=np.argmax(model_risk.predict(perprocess_data_risk(X)) )+1   #predict data
        return {"risk ide" : str(predict_risk)}
    else:
        return "bad request"
        

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
