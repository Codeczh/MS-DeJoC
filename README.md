# MS-DeJoC
This is the PyTorch implementation for our paper: **Decoupling Joint-Optimized Co-Teaching Strategy For Fine-Grained Visual Recognition**

## Network Architecture
The architecture of our proposed MS-DeJoC model is as follows
[![6PXw7Q.png](https://s3.ax1x.com/2021/03/01/6PXw7Q.png)](https://imgtu.com/i/6PXw7Q)
## Setups
 - **Environment**
After creating a virtual environment of python 3.7, run `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt` to install all dependencies.

 - **Data Preparation**
Download these web fine-grained datasets, namely [Web-CUB](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-bird.tar.gz), [Web-Car](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-car.tar.gz) and [Web-Aircraft](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-aircraft.tar.gz). Then uncompress them into `./data` directory.

## Running
According to your own demand, modify the corresponding parameters in `Main_ms_dejoc.py`, such as `--bs`, `--net`, `--data`, `--gpu`, etc. You can also directly run `Main_ms_dejoc.py` to get the final result.

## Results
[![6iQwtJ.png](https://s3.ax1x.com/2021/03/01/6iQwtJ.png)](https://imgtu.com/i/6iQwtJ)
