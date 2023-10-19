# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
import gradio
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
# ...
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def function(text, image):
    return text, image


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    gradio_app = gradio.Interface(fn=function, inputs=['text', 'image'], outputs=['text', 'image'])
    gradio_app.launch(share=False)
