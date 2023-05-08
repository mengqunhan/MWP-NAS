## Non-Autoregressive Math Word Problem Solver with Unified Tree Structure 

This is the offcial repo for the paper "[Non-Autoregressive Math Word Problem Solver with Unified Tree Structure]()".


### Requirements

* Pytorch = 1.13.1
* You can see the `requirements.txt` in each dictionary, run `pip install -r requirements.txt` to get the environment ready.


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
#
If you want to load the check points, please make sure you have downloaded the required files.
#
To use the check points and test the model on Math23K, run:
```
python ./Math23K/main.py --mode test
```
To use the check points and test the model on MAWPS, run:
```
python ./MAWPS/test_mawps.py
```
