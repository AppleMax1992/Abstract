from flask import Flask, render_template, jsonify,request
from random import *
from flask_cors import CORS
from T5_model import get_T5_result

app = Flask(__name__,
            static_folder = "./dist/static",
            template_folder = "./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/random')
def random_number():
    response = {
        'randomNumber': randint(1, 100)
    }
    return jsonify(response)
@app.route('/api/getInfo')
def getInfo():
    response = {
        'getInfo': '我是摘要'
    }
    return jsonify(response)
@app.route('/api/getT5',methods=['GET'])
def getT5():
    print("===============")
    content  = request.args.get('content')
    print(content)
    response = {
        'getInfo': get_T5_result(content)
    }
    print(request.args.get('numb'))
    return jsonify(response)

# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def catch_all(path):
#     if app.debug:
#         return requests.get('http://localhost:8080/{}'.format(path)).text
#     return render_template("index.html")


if __name__ == '__main__':
    print('======')
    app.run()

