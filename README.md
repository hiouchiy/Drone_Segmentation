# Segmentation using Aerial Drone Images
This repository provides a semantic segmentation model created using open data called [Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset).

It also introduces the source code for fine tuning based on the models and techniques to speed up the inference of the models.

## The definition of classes on the model
Since the open data "Semantic Drone Dataset" is a dataset consisting of 24 classes as shown below, this model classifies the same 24 classes.

| Name        | R   | G   | B   |
| ----------- | --- | --- | --- |
| unlabeled   | 0   | 0   | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/unlabeled.png" /></p>   |
| paved-area  | 128 | 64  | 128 | <p align="center"><img width = "30" height= "20" src="./label_colors/paved-area.png" /></p>  |
| dirt        | 130 | 76  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/dirt.png" /></p>        |
| grass       | 0   | 102 | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/grass.png" /></p>       |
| gravel      | 112 | 103 | 87  | <p align="center"><img width = "30" height= "20" src="./label_colors/gravel.png" /></p>      |
| water       | 28  | 42  | 168 | <p align="center"><img width = "30" height= "20" src="./label_colors/water.png" /></p>       |
| rocks       | 48  | 41  | 30  | <p align="center"><img width = "30" height= "20" src="./label_colors/rocks.png" /></p>       |
| pool        | 0   | 50  | 89  | <p align="center"><img width = "30" height= "20" src="./label_colors/pool.png" /></p>        |
| vegetation  | 107 | 142 | 35  | <p align="center"><img width = "30" height= "20" src="./label_colors/vegetation.png" /></p>  |
| roof        | 70  | 70  | 70  | <p align="center"><img width = "30" height= "20" src="./label_colors/roof.png" /></p>        |
| wall        | 102 | 102 | 156 | <p align="center"><img width = "30" height= "20" src="./label_colors/wall.png" /></p>        |
| window      | 254 | 228 | 12  | <p align="center"><img width = "30" height= "20" src="./label_colors/window.png" /></p>      |
| door        | 254 | 148 | 12  | <p align="center"><img width = "30" height= "20" src="./label_colors/door.png" /></p>        |
| fence       | 190 | 153 | 153 | <p align="center"><img width = "30" height= "20" src="./label_colors/fence.png" /></p>       |
| fence-pole  | 153 | 153 | 153 | <p align="center"><img width = "30" height= "20" src="./label_colors/fence-pole.png" /></p>  |
| person      | 255 | 22  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/person.png" /></p>      |
| dog         | 102 | 51  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/dog.png" /></p>         |
| car         | 9   | 143 | 150 | <p align="center"><img width = "30" height= "20" src="./label_colors/car.png" /></p>         |
| bicycle     | 119 | 11  | 32  | <p align="center"><img width = "30" height= "20" src="./label_colors/bicycle.png" /></p>     |
| tree        | 51  | 51  | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/tree.png" /></p>        |
| bald-tree   | 190 | 250 | 190 | <p align="center"><img width = "30" height= "20" src="./label_colors/bald-tree.png" /></p>   |
| ar-marker   | 112 | 150 | 146 | <p align="center"><img width = "30" height= "20" src="./label_colors/ar-marker.png" /></p>   |
| obstacle    | 2   | 135 | 115 | <p align="center"><img width = "30" height= "20" src="./label_colors/obstacle.png" /></p>    |
| conflicting | 255 | 0   | 0   | <p align="center"><img width = "30" height= "20" src="./label_colors/conflicting.png" /></p> |


## Getting Started
### Prerequisites
#### For On-prem environment
- Machine with Intel CPU (Core or Xeon)
    - Core: 10th generation or higher
    - Xeon: 2nd generation Xeon scalable processor or higher
- OS: Windows 10(WSL2) / Ubuntu 18.04 or later
- Docker (*Installation procedure described below)
#### For Google Colaboratory
- Win10/Mac/Linux PC with access to internet environment
- Google account
### Installing
#### Open host OS ports (only if accessing remotely)
This repo uses Jupyter Lab. In particular, if you will be accessing the server remotely, open port 8888 on the host OS in accordance with the procedures for each environment.
#### Install Docker
```Bash
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install -y docker-ce
sudo usermod -aG docker ${USER}
su - ${USER}
id -nG
```

#### Download Docker image
```Bash
docker pull continuumio/anaconda3
```
#### Launch Docker container
The container is started as the Root user. Also, port 8888 should be bound to the host OS and the container.
```Bash
sudo docker run -it -u 0 --privileged -p 8888:8888 continuumio/anaconda3 /bin/bash
```
Thereafter, the work will be done on the container.
#### Install additional software
```Bash
apt-get update
apt-get install -y wget unzip git sudo vim numactl
apt-get install -y libgl1-mesa-dev
conda create -n ov python=3.8 -y
conda activate ov
pip install --upgrade pip
pip install torch torchvision torchaudio pandas scikit-learn statistics pillow opencv-python albumentations tqdm matplotlib typing-extensions==4.4.0 jupyterlab segmentation-models-pytorch torchsummary
pip install ipywidgets widgetsnbextension
```
#### Clone this repo
```Bash
cd ~
git clone https://github.com/hiouchiy/Drone_Segmentation.git
```
#### Launch Jupyter Lab
```Bash
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --port=8888
```
#### Access the Jupyter lab from Web browser(MS Edge or Chrome)
After executing the previous command, you should see the following log output, with the URL with a token for the local host (127.0.0.1) at the end. Paste this URL into your web browser to access the site. If you are accessing remotely, change the IP address to the IP address of the server's host OS.
```
root@f79f54d47c1b:~# jupyter lab --allow-root --ip=0.0.0.0 --no-browser
[I 09:13:08.932 LabApp] JupyterLab extension loaded from /usr/local/lib/python3.6/dist-packages/jupyterlab
[I 09:13:08.933 LabApp] JupyterLab application directory is /usr/local/share/jupyter/lab
[I 09:13:08.935 LabApp] Serving notebooks from local directory: /root
[I 09:13:08.935 LabApp] Jupyter Notebook 6.1.4 is running at:
[I 09:13:08.935 LabApp] http://f79f54d47c1b:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3
[I 09:13:08.935 LabApp]  or http://127.0.0.1:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3
[I 09:13:08.935 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 09:13:08.941 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-33-open.html
    Or copy and paste one of these URLs:
        http://f79f54d47c1b:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3
     or http://127.0.0.1:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3
```
↑For example in above log, the last row is showing the URL to Jupyter lab. "http://127.0.0.1:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3" です。
#### Open a notebook in Jupyter lab
Open the notebook of your choice from the list below in Jupyter Lab, and then follow the contents of the notebook.

- TrainModel.ipynb: 
Train a model with 
[Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)」
- Test_PreTrainModel.ipynb: 
Infer images with [the trained model](https://drive.google.com/file/d/14PtYuFZc-5sB2n9lLUDku8bgyEKSLZG5/view?usp=share_link).
- Fine-Tuning.ipynb: 
Fine tune [our trained model](https://drive.google.com/file/d/14PtYuFZc-5sB2n9lLUDku8bgyEKSLZG5/view?usp=share_link) with custom dataset.
- Test_CustumModel.ipynb: 
Infer images with [the fine tuned model](https://drive.google.com/file/d/1JXPHg4brau1T93z79VNr4VqLeCEx2CcW/view?usp=share_link).
- Model_Optimization_and_Quantization.ipynb: 
Some techniques to accelerate model inference on CPU.
##### Hint to run correctly
Rewrite the notebook path setting to any path and run from the top of the notebook
- IMAGE_PATH, MASK_PATH: Path to the folder containing the image data and annotated images for model training
- MODEL_PATH: Path to the pre-trained model files


## License
This project is licensed under the MIT License.

## Acknowledgments
This source code is based on [here](https://github.com/G21TKA01/Drone_Segmentation) with some arrangements. The author, [G21TKA01](https://github.com/G21TKA01), has given his permission to use this code in advance. Once again, we would like to thank the author for providing us with a wonderful code.