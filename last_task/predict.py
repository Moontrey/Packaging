from sklearn.model_selection import cross_val_predict
import pickle
import traceback
from flask import Flask, request, jsonify
import ast
import numpy as np
import sys

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

clf = pickle.load(open('model', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        dic = request.get_data().decode('ISO-8859-1')
        print(dic)
        dic = ast.literal_eval(dic)
        print(dic)
        X = np.array(list(dic['features']))
        yhat = clf.predict(X.reshape(1, -1))
        print(type(yhat))
        return jsonify({'prediction': yhat.tolist()})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80
    app.run(host='0.0.0.0', port=port, debug=True)

# to run at prot 777
# python last_task/predict.py 7777

# then open a terminal
# curl http://0.0.0.0:7777/predict -H 'Content-Type: application/json' -d '{"features": [0.0, 4925.0, 1.95406,  79.0, 3871.0]}'