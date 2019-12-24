import argparse
import keras
import numpy as np
import tensorflow as tf

from keras_utils import construct_hrs_model
from project_utils import get_data
from block_split_config import get_split


def defend_adversarial_attack(dataset, split, model_indicator, attack, epsilon, test_samples, num_steps, step_size,
                              attack_setting, gradient_samples):
    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # get block definitions
    blocks_definition = get_split(split, dataset)

    # construct model
    keras.backend.set_learning_phase(0)
    model = construct_hrs_model(dataset=dataset, model_indicator=model_indicator, blocks_definition=blocks_definition)

    # get data
    [X_train, X_test, Y_train, Y_test] = get_data(dataset=dataset, scale1=True, one_hot=False, percentage=0.01)

    # make attack object
    if attack == 'FGSM':
        from attack_utils import FGSM
        attack = FGSM(model=model, epsilon=epsilon, dataset=dataset)
    elif attack == 'PGD':
        from attack_utils import PGD
        attack = PGD(model=model, num_steps=num_steps, step_size=step_size, epsilon=epsilon, dataset=dataset)
    elif attack == 'CWPGD':
        from attack_utils import CW_PGD
        attack = CW_PGD(model=model, num_steps=num_steps, step_size=step_size, epsilon=epsilon, dataset=dataset)
    else:
        raise ValueError('%s is not a valid attack name!' % attack)

    # perform attack
    result = []
    distortion = []

    for test_sample_idx in range(test_samples):
        print('generating adv sample for test sample ' + str(test_sample_idx))
        image = X_test[test_sample_idx:test_sample_idx + 1]
        label = Y_test[test_sample_idx:test_sample_idx + 1]

        for target in range(10):
            if target == label:
                continue

            target_input = np.array([target])
            if attack_setting == 'normal':
                adversarial = attack.perturb(image, target_input, sess)
            elif attack_setting == 'EOT':
                adversarial = attack.perturb_gm(image, target_input, sess, gradient_samples=gradient_samples)
            else:
                raise ValueError('%s is not a valid attack setting!' % attack_setting)

            output = model.predict(adversarial)
            adv_pred = np.argmax(list(output)[0])
            result.append((adv_pred == target).astype(int))

            l_inf = np.amax(adversarial - image)
            distortion.append(l_inf)

    # compute attack success rate (ASR) and average distortion(L_inf)
    succ_rate = np.array(result).mean()
    mean_distortion = np.array(distortion).mean()

    print('Perform %s attack to model %s' % (attack, model_indicator))
    print('Attack succ rate (ASR) = %.4f' % succ_rate)
    print('Average distortion = %.2f' % mean_distortion)


if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[10][10]', help='model indicator, format: model_name[5][5] for'
                                                                            'a HRS model with 5 by 5 channels')
    parser.add_argument('--split', default='default', help='the indicator of channel structures in each block')
    parser.add_argument('--dataset', default='CIFAR', help='CIFAR or MNIST')
    parser.add_argument('--test_examples', default=10, help='number of test examples')
    parser.add_argument('--attack', default='CWPGD', help='FGSM, PGD or CWPGD')
    parser.add_argument('--epsilon', default=8, help='the L_inf bound of allowed adversarial perturbations, 8 means 8/255',
                        type=float)
    parser.add_argument('--num_steps', default=100, help='number of steps in generating adversarial examples, not work '
                                                         'for FGSM')
    parser.add_argument('--step_size', default=0.1, help='the step size in generating adversarial examples')
    parser.add_argument('--attack_setting', default='normal', help='normal or EOT')
    parser.add_argument('--gradient_samples', default=10, help='number of gradient samples for calculating gradient expectation.'
                                                               ', only work when --attack_setting is set to EOT.')

    args = parser.parse_args()
    defend_adversarial_attack(dataset=args.dataset,
                              split=args.split,
                              model_indicator=args.model_indicator,
                              attack=args.attack,
                              epsilon=args.epsilon,
                              test_samples=args.test_examples,
                              num_steps=args.num_steps,
                              step_size=args.step_size,
                              attack_setting=args.attack_setting,
                              gradient_samples=args.gradient_samples
                              )
