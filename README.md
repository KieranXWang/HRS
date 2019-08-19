Hierarchical Random Switching
=============================
Hierarchical Random Switching (HRS) is a stochastic defense technique which
converts a deterministic base model structure to a stochastic model in
order to protect the model from adversarial threats such as adversarial
(mis-classification) attack and adversarial reprogramming. The
figure below illustrates the structure of a HRS model.

![](https://github.com/KieranXWang/HRS/raw/master/Figures/ijcai_hrs.png)

In the inference phase, the active channel that processes the input of
each block is randomly assigned and ever-switching, leading to drastic but
unpredictable changes to the active path (the chain of activate channels
which has the same architecture as the base model).

For more details, please refer to our paper at IJCAI 2019. The link of
appendices of the paper is .

If you have any questions, feel free to contact Xiao Wang by email kxw@bu.edu.


## Set Up
This code requires `python 3` and the following packages: `tensorflow`,
`keras`, `numpy`.

This code is tested with .


## Train HRS


## Compute Test Accuracy

## Defense against Adversarial Attack

## Defense against Adversarial Reprogramming

