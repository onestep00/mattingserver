# mattingserver
image segmentation &amp; matting server


U2Net - https://github.com/NathanUA/U-2-Net  
FBA_Matting - https://github.com/MarcoForte/FBA_Matting  
GCA_Matting - https://github.com/Yaoyi-Li/GCA-Matting


## Getting Started

Tested only at Colab

### Prerequisites

What things you need to install the software and how to install them

```
flask_ngrok
toml
tensorboardX
```

You need to prepare a trained model.


## Running the tests

start server
```
python main.py

```
options

```
--u2 : u2net params(.pth)
--u2-size : u2net image input size(foursquare)
--gca : gca params(.pth) 
--gca-config : gca-config file(.toml)

```
