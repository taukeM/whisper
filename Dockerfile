FROM python:3
COPY requirements.txt /
RUN pip install --default-timeout=100 -r requirements.txt 
RUN pip install --upgrade accelerate
COPY train.py /
CMD [ "python", "./train.py" ]
