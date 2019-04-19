from flask import Flask
from flask import jsonify
from model.hzy.lstm.release import classifier
app = Flask(__name__)


@app.route('/')
def LSTM():
    # automatically using test data to perform classification
    model_path = './model/hzy/lstm/model_saved/'
    data_path = './model/hzy/lstm/dataset/'
    result, prediction = classifier(model_path, data_path)
    return jsonify(result)


if __name__ == '__main__':
    app.run()
