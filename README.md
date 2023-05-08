## Non-Autoregressive Math Word Problem Solver with Unified Tree Structure 

This is the offcial repo for the paper "[Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction](https://arxiv.org/abs/2203.10316)".


### Requirements

* Pytorch = 1.13.1
* You can see the `requirements.txt` in each dictionary, run `pip install -r requirements.txt` to get environment ready.


### Usage

#### train
To train our model in Math23K, run:
```
python ./Math23K/main.py --mode train
```
To train our model in MAWPS, run:
```
python ./MAWPS/main_mawps.py
```
#### test
#
If you want to load the check points, please make sure you have downloaded the required files.
#
To use the check points and test the model in Math23K, run:
```
python ./Math23K/main.py --mode test
```
To use the check points and test the model in MAWPS, run:
```
python ./MAWPS/test_mawps.py
```
