# Build this image with name:
# ${HUB_DOMAIN}/mlmodel/object_detection:${REST_TAG}
ARG HUB_DOMAIN="swr.cn-north-4.myhuaweicloud.com"
FROM ${HUB_DOMAIN}/mlmodel/restful_cuda_ubuntu18.04:latest

ENV PIPURL "https://pypi.douban.com/simple/"

# virtual env
WORKDIR /venv
RUN virtualenv prod
ENV PATH="/venv/prod/bin:$PATH"
RUN /bin/bash -c "source ./prod/bin/activate"
WORKDIR /codes

# pip
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i ${PIPURL}

COPY . .
RUN chmod +x run.sh
CMD bash run.sh