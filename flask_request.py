# 启用flask_start的服务后，将数据以post的方式调用服务得到结果
import os
import json
import base64
import requests


def image_encode(image_path):
    with open(image_path, 'rb')as f:
        image_byte = f.read()
    image_base64 = base64.b64encode(image_byte)
    image = image_base64.decode()
    return image


if __name__ == '__main__':
    url = 'http://0.0.0.0:9999/test/'  # 根据flask_start中的设置: http://host:port/name/
    path_dir = 'image'
    path_list = os.listdir(path_dir)
    for image_path in path_list:
        image_path = f'{path_dir}/{image_path}'
        image = image_encode(image_path)
        request_dict = {'image': image}
        request = json.dumps(request_dict)
        response = requests.post(url, data=request)
        result = response.json()
        print(result)
