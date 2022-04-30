import datetime
import json
import os
import re
from random import random

import matplotlib.pyplot as plt
import pymysql
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from torchvision import transforms

from grad_cam_evaluate_single import explainInGradCAM
from model import resnet50

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pymysql.install_as_MySQLdb()

SECRET_KEY = 'abcdefghijklmm'
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:lsyshxx674@localhost:3306/testbase?charset=utf8'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# 注册
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = json.loads(request.data)
        print(data)
        username = data['username']
        password = data['password']
        password2 = data['password2']
        mobile = data['mobile']
        email = data['email']
        identity = data['identity']
        school = data['school']
        login_time = str(datetime.datetime.now()).split('.')[0]

        if not all([username, password, password2, mobile, email, identity, school]):
            return jsonify({
                "meta": {
                    "msg": "参数不完整",
                    "status": 400
                }})
        elif password != password2:
            return jsonify({
                "meta": {
                    "msg": "两次密码不一致，请重新输入",
                    "status": 400
                }})
        else:
            new_user = Users(username=username, password=password,
                             id=random(), rid=random(), mobile=mobile, email=email,
                             identity=identity, school=school, login_time=login_time)
            db.session.add(new_user)
            db.session.commit()
            return jsonify({
                "data": {
                    "msg": "注册成功"
                },
                "meta": {
                    "msg": "注册成功",
                    "status": 200
                }})
    return render_template('register.html')


# 登录
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        data = json.loads(request.data)
        username = data['username']
        password = data['password']
        user_id = verify_auth_token(username)
        if not all([username, password]):
            return jsonify({
                "meta": {
                    "msg": "请输入用户名和密码",
                    "status": 400
                }})
        user = Users.query.filter(
            Users.username == username, Users.password == password).first()
        if user:
            user = Users.query.filter(Users.username == username).first()
            success = {"data": {
                "id": user.id,
                "rid": user.rid,
                "username": user.username,
                "mobile": user.mobile,
                "email": user.email,
                "identity": user.identity,
                "school": user.school,
                "login_time": user.login_time,
                "token": generate_auth_token(user.username)
            },
                "meta": {
                    "msg": "获取成功",
                    "status": 200
                }}
            return jsonify(success)
        else:
            return jsonify({
                "meta": {
                    "msg": "用户名或密码错误",
                    "status": 400
                }
            })


# 上传图片
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img = request.files['file']
        img = Image.open(img)
        img.save("static/yuantu.jpg")
        plt.imshow(img)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(
            json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = resnet50(num_classes=79).to(device)

        # load model weights
        weights_path = "./resnet50.pth"
        assert os.path.exists(
            weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        # prediction
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        prob = "{:.3}".format(predict[predict_cla].numpy())
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        plt.title(print_res)
        print_res = {'painter': class_indict[str(predict_cla)], 'prob': prob}
        plt.savefig("./static/fenleitu.png")
        explainInGradCAM("static/yuantu.jpg")
        return jsonify(print_res)
    else:
        return render_template('upload.html')


# 定义一个用户及密码的数据库
class Users(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    rid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(10))
    mobile = db.Column(db.String(20), primary_key=True)
    email = db.Column(db.String(50), primary_key=True)
    token = db.Column(db.Integer)
    password = db.Column(db.String(16))
    identity = db.Column(db.String(16))
    school = db.Column(db.String(50))
    login_time = db.Column(db.String(50))


def generate_auth_token(user_id, expiration=36000):
    s = Serializer(SECRET_KEY, expiration)
    return str(s.dumps({'user_id': user_id}), encoding='utf-8')


def verify_auth_token(username):
    user_id = re.sub(r'^"|"$', '', username)
    return user_id


# db.drop_all()
# db.create_all()

if __name__ == '__main__':
    app.run()
