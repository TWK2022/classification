# pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用flask将程序包装成一个服务，并在服务器上启动
import cv2
import json
import flask
import base64
import argparse
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动flask服务|')
# ...
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
app = flask.Flask(__name__)  # 创建一个服务框架


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def image_decode(image):
    image_base64 = image.encode()  # base64
    image_byte = base64.b64decode(image_base64)  # base64->字节类型
    array = np.frombuffer(image_byte, dtype=np.uint8)  # 字节类型->一行数组
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)  # 一行数组->BGR图片
    return image


@app.route('/test/', methods=['POST'])  # 每当调用服务时会执行一次flask_app函数
def flask_app():
    request_json = flask.request.get_data()
    request_dict = json.loads(request_json)
    image = image_decode(request_dict['image'])
    # ...
    result = image.shape
    return result


if __name__ == '__main__':
    print('| 使用flask启动服务 |')
    app.run(host='0.0.0.0', port=9999, debug=False)  # 启动服务
