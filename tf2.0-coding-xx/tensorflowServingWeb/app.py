import flask
from flask import request,jsonify
import werkzeug
import os
import tensorflow as tf
import getConfig
import numpy as np
import pickle
import requests
import json
from PIL import Image
gConfig = {}
gConfig = getConfig.get_config(config_file='config.ini')

app = flask.Flask("imgClassifierWeb")

def CNN_predict():
        global secure_filename
        file = gConfig['dataset_path'] + "batches.meta"
        patch_bin_file = open(file, 'rb')
        label_names_dict = pickle.load(patch_bin_file)["label_names"]
        img = Image.open(os.path.join(app.root_path, secure_filename))
        img = img.convert("RGB")
        r, g, b = img.split()
        r_arr = np.array(r)
        g_arr = np.array(g)
        b_arr = np.array(b)
        img = np.concatenate((r_arr, g_arr, b_arr))

        image = img.reshape([1, 32, 32, 3])/255
        payload = json.dumps({"instances":image.tolist()})

        predicted_class=requests.post('http://localhost:9000/v1/models/ImageClassifier:predict',data=payload)

        predicted_class=np.array(json.loads(predicted_class.text)["predictions"])

        prediction=tf.math.argmax(predicted_class[0]).numpy()
        print(prediction)
        
        return flask.render_template(template_name_or_list="prediction_result.html",predicted_class=label_names_dict[prediction])

app.add_url_rule(rule="/predict/", endpoint="predict", view_func=CNN_predict)

@app.route('/api', methods=['POST'])
def predict_api():
    file = gConfig['dataset_path'] + "batches.meta"
    patch_bin_file = open(file, 'rb')
    label_names_dict = pickle.load(patch_bin_file)["label_names"]
    img = Image.open( request.form['path'])
    img = img.convert("RGB")
    r, g, b = img.split()
    r_arr = np.array(r)
    g_arr = np.array(g)
    b_arr = np.array(b)
    img = np.concatenate((r_arr, g_arr, b_arr))

    image = img.reshape([1, 32, 32, 3]) / 255
    payload = json.dumps({"instances": image.tolist()})

    predicted_class = requests.post('http://localhost:9000/v1/models/ImageClassifi:predict', data=payload)

    predicted_class = np.array(json.loads(predicted_class.text)["predictions"])

    prediction = tf.math.argmax(predicted_class[0]).numpy()
    print(prediction)
    return jsonify({'result':label_names_dict[prediction]})





"""
知识点：
flask.request属性
form: 
一个从POST和PUT请求解析的 MultiDict（一键多值字典）。
args: 
MultiDict，要操作 URL （如 ?key=value ）中提交的参数可以使用 args 属性:
searchword = request.args.get('key', '')
values: 
CombinedMultiDict，内容是form和args。 
可以使用values替代form和args。
cookies: 
顾名思义，请求的cookies，类型是dict。
stream: 
在可知的mimetype下，如果进来的表单数据无法解码，会没有任何改动的保存到这个·stream·以供使用。很多时候，当请求的数据转换为string时，使用data是最好的方式。这个stream只返回数据一次。
headers: 
请求头，字典类型。
data: 
包含了请求的数据，并转换为字符串，除非是一个Flask无法处理的mimetype。
files: 
MultiDict，带有通过POST或PUT请求上传的文件。
method: 
请求方法，比如POST、GET
知识点参考链接：https://blog.csdn.net/yannanxiu/article/details/53116652
werkzeug
"""

def upload_image():
    global secure_filename
    if flask.request.method == "POST":  # 设置request的模式为POST
        img_file = flask.request.files["image_file"]  # 获取需要分类的图片
        secure_filename = werkzeug.secure_filename(img_file.filename)  # 生成一个没有乱码的文件名
        img_path = os.path.join(app.root_path, secure_filename)  # 获取图片的保存路径
        img_file.save(img_path)  # 将图片保存在应用的根目录下
        print("图片上传成功.")
        """

        """
        return flask.redirect(flask.url_for(endpoint="predict"))
    return "图片上传失败"


"""
"""
app.add_url_rule(rule="/upload/", endpoint="upload", view_func=upload_image, methods=["POST"])


def predirect_upload():
    return flask.render_template(template_name_or_list="upload_image.html")

"""
"""
app.add_url_rule(rule="/", endpoint="homepage", view_func=predirect_upload)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008, debug=False)
