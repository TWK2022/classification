# pip install gunicorn -i https://pypi.tuna.tsinghua.edu.cn/simple
# 使用gunicorn启用flask：gunicorn -c gunicorn_config.py flask_start:app
# 设置端口，外部访问端口也会从http://host:port/name/变为http://bind/name/
bind = '0.0.0.0:9999'
# 设置进程数。推荐核数*2+1发挥最佳性能
workers = 3
# 客户端最大连接数，默认1000
worker_connections = 2000
# 设置工作模型。有sync(同步)(默认)、eventlet(协程异步)、gevent(协程异步)、tornado、gthread(线程)。
# sync根据请求先来后到处理。eventlet需要安装库：pip install eventlet。gevent需要安装库：pip install gevent。
# tornado需要安装库：pip install tornado。gthread需要指定threads参数
worker_class = 'sync'
# 设置线程数。指定threads参数时工作模式自动变成gthread(线程)模式
threads = 1
# 启动程序时的超时时间(s)
timeout = 60
# 当代码有修改时会自动重启，适用于开发环境，默认False
reload = True
# 设置日志的记录地址。需要提前创建gunicorn_log文件夹
accesslog = 'gunicorn_log/access.log'
# 设置错误信息的记录地址。需要提前创建gunicorn_log文件夹
errorlog = 'gunicorn_log/error.log'
# 设置日志的记录水平。有debug、info(默认)、warning、error、critical，按照记录信息的详细程度排序
loglevel = 'info'
