# 智库图像分类模型

## 简介
* 智库图像分类模型
* 目前有以下几个模型

    |模型名|英文名|备注|
    |  ----  | ----  | ---- |
    | 暴恐 | terror    ||
    | 色情补充 | eroticext ||



## 使用方式

### DOCKER

* 宿主机需提供的环境变量

  | 变量类别         | 变量名                | 备注                             |
  | ---------------- | --------------------- | -------------------------------- |
  | 模型镜像仓库地址 | HUB_DOMAIN            | swr.cn-north-4.myhuaweicloud.com |
  | 注册中心IP地址   | ETCD_HOST             |                                  |
  | 宿主机IP地址     | SERVICE_HOST          |                                  |
  | RESTFUL服务tag   | REST_TAG              | 1.0.0                            |
  | 模型版本号       | EROTICEXT_CLS_VERSION | 如: v_100, 每个模型都需指定      |
  |                  | TERROR_CLS_VERSION    |                                  |
  
* 创建基础镜像(或者从仓库上面拉取)
  `cd engine_restful && sudo docker build -t ${HUB_DOMAIN}/mlmodel/restful_cuda_ubuntu18.04:latest -f Dockerfile . && cd ..`

* 创建日志卷

  `sudo docker volume create app_logs`


* 推荐使用`docker-compose`方式启动容器

​		`sudo -E docker-compose -f docker-compose.yml up -d`