FROM tiangolo/uvicorn-gunicorn:python3.8-slim 

RUN apt-get update && apt-get install gcc ffmpeg libsm6 libxext6 -y

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --upgrade pip

RUN pip install torch==1.10.1+cpu torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

RUN pip install -r requirements.txt
RUN mkdir .pem
COPY path/to/file.json ./.pem/ptvn-vision-iron-wave-143203-ea26bbebdfe7.json

ENV GOOGLE_SERVICE_ACCOUNT_PATH=".pem/ptvn-vision-iron-wave-143203-ea26bbebdfe7.json"

COPY deepdoctection deepdoctection
COPY models models
COPY app.py .

ENTRYPOINT ["python"]
EXPOSE 8080

CMD ["app.py"]
