## pytorch图片分类训练框架
>代码兼容性较强，使用的是一些基本的库、基础的函数  
>在argparse中可以选择使用wandb，能在wandb网站中生成可视化的训练过程
### 1，环境
>torch：https://pytorch.org/get-started/previous-versions/
>```
>pip install timm tqdm wandb opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
### 2，数据格式
>├── 数据集路径：data_path  
>&emsp; &emsp; └── image：存放所有图片  
>&emsp; &emsp; └── train.txt：训练图片的绝对路径(或相对data_path下路径)和类别号，  
>&emsp; &emsp; &emsp; &emsp; (如-->image/mask/0.jpg 0 2<--表示该图片类别为0和2，空类别图片无类别号)  
>&emsp; &emsp; └── val.txt：验证图片的绝对路径(或相对data_path下路径)和类别号  
>&emsp; &emsp; └── class.txt：所有的类别名称  
### 3，run.py
>模型训练时运行该文件，argparse中有对每个参数的说明
### 4，predict_pt.py
>使用训练好的pt模型预测
### 5，export_onnx.py
>将pt模型导出为onnx模型
### 6，predict_onnx.py
>使用导出的onnx模型预测
### 7，export_trt_record
>文档中有onnx模型导出为tensort模型的详细说明
### 8，predict_trt.py
>使用导出的trt模型预测
### 9，gradio_start.py
>用gradio将程序包装成一个可视化的页面，可以在网页可视化的展示
### 10，flask_start.py
>用flask将程序包装成一个服务，并在服务器上启动
### 11，flask_request.py
>以post请求传输数据调用服务
### 12，gunicorn_config.py
>用gunicorn多进程启动flask服务：gunicorn -c gunicorn_config.py flask_start:app
### 其他
>学习笔记：https://github.com/TWK2022/notebook
