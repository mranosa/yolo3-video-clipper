# yolo3-video-clipper

Pre-Requisite
=========
- Install ["Conda"](https://conda.io/docs/)
- Download a video with a car and save it as `video.avi`

Setup Commands
=========
### Linux
```sh
$ wget https://pjreddie.com/media/files/yolov3.weights
$ conda create -n vid-clipper-py35 python=3.5 anaconda
$ source activate vid-clipper-py35
$ conda install -c soumith pytorch
$ conda install -c conda-forge ffmpeg
$ conda install -c conda-forge opencv
```

Run Commands
=========
### Linux
```sh
$ python video.py
```
- Open `output.avi` for results


TODO
=========
 - Clip to multiple outputs
 - Name `output.avi` to `{time-start}.avi`? (To tell start time)