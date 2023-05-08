## Non-Autoregressive Math Word Problem Solver with Unified Tree Structure 

This is the offcial repo for the paper "[Non-Autoregressive Math Word Problem Solver with Unified Tree Structure]()".


### Requirements

* Pytorch = 1.13.1;
* You can see the `requirements.txt` in each dictionary, run `pip install -r requirements.txt` to get environment ready.

### Datasets

Before you use the code, please download the missing train dataset from this [link](https://pan.baidu.com/s/17t6NZUjDW9MJdi0UzK9TJg?pwd=q61p) and put it in `datasets` dictionary.

### Usage

#### Train
To train our model on Math23K, run:
```
python ./Math23K/main.py --mode train
```
To train our model on MAWPS, run:
```
python ./MAWPS/main_mawps.py
```
#### Test

If you want to load the check points, please make sure you have downloaded the required files:
* [Math23K check points](https://pan.baidu.com/s/1CQgW2mV3Mt7ry6gKIw9j3Q?pwd=28iz)
* [MAWPS check points](https://pan.baidu.com/s/1_tTEemyCXZQ0H4gkFOuqJg?pwd=pzy1)

Please put the check point files(Math23K or MAWPS) in `check_points` dictionary and put the dictionary in Math23K or MAWPS dictionary.

To use the check points and test the model on Math23K, run:
```
python ./Math23K/main.py --mode test
```
To use the check points and test the model on MAWPS, run:
```
python ./MAWPS/test_mawps.py
```
#### Citation
#
If this repo is helpful to your work, please kindly cite our paper:
```
@article{mwpnas2023,
  title={Non-Autoregressive Math Word Problem Solver with Unified Tree Structure},
  author={Bin, Yi and Han, Mengqun and Shi, wenhao and Wang, Lei and Yang, Yang and Shen, Heng Tao},
  journal={arXiv},
  year={2023}
}
```
