# whisper

## Create a dataset
```
$ python3 dataset.py
```

## Build a docker image
```
docker build -t {dockerhub repo}:latest .
```

## Run the docker image with mounted volumes
```
sudo docker run -v $(pwd)/Dataset:/Dataset -v $(pwd)/model:/model {docker image}
```
