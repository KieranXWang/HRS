Hierarchical Random Switching
=============================
2019.09.23:
It is more than welcome to test HRS against other adversarial evasion attacks and other types of adversarial threats. If you would like to have pretained HRS models to accelerate your research, we are happy to provide them and just contact me by sending emails to  kxw@bu.edu.

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

For more details, please refer to our paper https://arxiv.org/abs/1908.07116#. And cite our paper using:
`
@article{Wang2019ProtectingNN,
  title={Protecting Neural Networks with Hierarchical Random Switching: Towards Better Robustness-Accuracy Trade-off for Stochastic Defenses},
  author={Xiao Wang and Siyue Wang and Pin-Yu Chen and Yanzhi Wang and Brian Kulis and Xue Lin and Sang Peter Chin},
  journal={ArXiv},
  year={2019},
  volume={abs/1908.07116}
}
`


If you have any questions, feel free to contact Xiao Wang by email kxw@bu.edu.


## Dependencies
This code requires `python 3` and the following packages: `tensorflow`,
`keras`, `numpy`.

This code is tested with `tensorflow 1.14.0`, `keras 2.2.4` and `numpy 1.16.4`.


## Train HRS
`python train_hrs.py [options]`

Options:
* `-- model_indicator`: the indicator of the trained model, which indicates
the HRS structure and will be used as the locator for retrieving trained model
weights. Format: `test_model[10][10]` for a two-block, 10 by 10 HRS model.
Default: `test_hrs[10][10]`.
* `--split`: the indicator of channel structures in each block.
Default: `default`, in this splitting, all convolutional layers and the first
fully-connected layer are grouped as the first block, the second fully-connected layer
and the output layer are grouped as the second block.
* `--train_schedule`: number of epochs for training each block. Default: `40 40`.
* `--dataset`: CIFAR or MNIST. Default: `CIFAR`.

Outputs:
Trained weights of each channel will be saved in `./Model/`.

#### Customize Model Structure and Block Splitting
This can be done by adding options in `block_split_config.py` with unique indicators. Note that `get_split` needs to
return a list of functions that return Keras `Sequential` models for each block.

## Compute Test Accuracy
`python test_acc.py [options]`

Options:
* `--model_indicator`: the indicator of the trained model (which is specified in training).
Default: `test_hrs[10][10]`.
* `--split`: the indicator of channel structures in each block. Default: `default`.
* `--dataset`: CIFAR or MNIST. Default: `CIFAR`.

Outputs:
Test accuracy of the specified HRS model will be printed. Note: because
of the randomness of model structure, different runs may result in slightly
different results.

## Defense against Adversarial Attack
`python defense_adversarial_attack.py [options]`

Options:
* `--model_indicator`: the indicator of the trained model (which is specified in training). Default: `test_hrs[10][10]`.
* `--split`: the indicator of channel structures in each block. Default: `default`.
* `--dataset`: CIFAR or MNIST. Default: `CIFAR`.
* `--test_examples`: number of test examples. Default: `10`.
* `--attack`: FGSM, PGD or CWPGD. Default: `CWPGD`.
* `--epsilon`: the L_inf bound of allowed adversarial perturbations. Default: `8/255`.
* `--num_steps`: number of steps in generating adversarial examples, not work for FGSM. Default: `100`.
* `--step_size`: the step size in generating adversarial examples. Default: `0.1`.
* `--attack_setting`: normal or EOT. Default: `normal`.
* `--gradient_samples`: number of gradient samples for calculating gradient expectation, only work when --attack_setting is set to EOT. Default: `10`.

Outputs:
Attack success rate (ASR) and averaged distortion will be printed.

## Defense against Adversarial Reprogramming
In this experiment we reprogram a trained CIFAR-10 classifier to do MNIST
classification by training input and output transformations. The input transformation
is performed by a locally-connected layer and the output transformation is
an identical mapping. The reprogramming test accuracy indicates the defense
capability against adversarial reprogramming: the lower the better.

`python defense_adversarial_reprogramming.py [options]`

Options:
* `--model_indicator`: the indicator of the trained model (which is specified in training). Default: `test_hrs[10][10]`.
* `--split`: the indicator of channel structures in each block. Default: `default`.
* `--epochs`: the number of epochs to train (reprogram). Default: `50`.

Outputs:
The reprogramming training and testing accuracies will be printed.

