#base image...alpine es lo basico de lo basico en linux
FROM python:alpine3.8
# copio lo local a /app
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./index.py



