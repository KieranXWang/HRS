import tensorflow as tf
import numpy as np


# FGSM
class FGSM(object):
    def __init__(self, model, epsilon,  dataset='CIFAR'):
        self.model = model
        self.epsilon = epsilon

        if dataset == 'CIFAR':
            self.height = 32
            self.width = 32
            self.channel = 3
        elif dataset == 'MNIST':
            self.height = 28
            self.width = 28
            self.channel = 1
        self.xs = tf.Variable(np.zeros((1, self.height, self.width, self.channel), dtype=np.float32), name='modifier')
        self.xs_place = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.xs_orig = tf.Variable(np.zeros((1, self.height, self.width, self.channel), dtype=np.float32), name='original')
        self.ys = tf.placeholder(tf.int32, [None])
        self.y_variable = tf.Variable(np.zeros((1,), dtype=np.int32), name='target')
        # assign operations
        self.assign_x = tf.assign(self.xs, self.xs_place)
        self.assign_x_orig = tf.assign(self.xs_orig, self.xs_place)
        self.assign_y = tf.assign(self.y_variable, self.ys)
        # clip operation
        self.do_clip_xs = tf.clip_by_value(self.xs, 0, 1)
        # logits
        self.logits = model(self.xs)
        # loss
        y_one_hot = tf.one_hot(self.y_variable, 10)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits , labels=y_one_hot)
        loss = tf.reduce_mean(loss)
        # Define gradient of loss wrt input
        self.grad = tf.gradients(loss, self.xs)[0]
        self.grad_sign = tf.sign(self.grad)

    def perturb(self, x, y, sess):
        # initialize
        sess.run(self.xs.initializer)
        sess.run(self.y_variable.initializer)
        sess.run(self.xs_orig.initializer)
        # assign
        sess.run(self.assign_x, feed_dict={self.xs_place: x})
        # sess.run(self.assign_x_orig, feed_dict={self.xs_orig: x})
        sess.run(self.assign_y, feed_dict={self.ys: y})

        # generate adv example
        #grad = sess.run(self.grad)
        grad_sign = sess.run(self.grad_sign)
        adv_x = x - self.epsilon * grad_sign

        return adv_x

    def sample_gradient(self, x, y, sess, gradient_samples):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        grad_list = np.zeros((gradient_samples, 1, self.height, self.width, self.channel))
        for grad_idx in range(gradient_samples):
            grad = sess.run(self.grad, feed_dict={self.ys: y})
            grad_list[grad_idx, :, :, :, :] = grad

        return grad_list

    def perturb_gm(self, x, y, sess, gradient_samples):
        # initialize
        sess.run(self.xs.initializer)
        sess.run(self.y_variable.initializer)
        sess.run(self.xs_orig.initializer)
        # assign
        sess.run(self.assign_x, feed_dict={self.xs_place: x})
        # sess.run(self.assign_x_orig, feed_dict={self.xs_place: x})
        sess.run(self.assign_y, feed_dict={self.ys: y})

        grad_list = np.zeros((gradient_samples, 1, self.height, self.width, self.channel))
        for i in range(gradient_samples):
            grad = sess.run(self.grad)
            grad_list[i, :, :, :, :] = grad
        grad_mean = np.mean(grad_list, axis=0, keepdims=False)
        grad_mean_sign = np.sign(grad_mean)
        adv_x = x - self.epsilon * grad_mean_sign

        return adv_x


# PGD
class PGD(object):
    def __init__(self, model, num_steps, step_size, epsilon, dataset='CIFAR'):
        if dataset == 'CIFAR':
            self.height = 32
            self.width = 32
            self.channel = 3
        elif dataset == 'MNIST':
            self.height = 28
            self.width = 28
            self.channel = 1
        elif dataset == 'Fashion_MNIST':
            self.height = 28
            self.width = 28
            self.channel = 1
        self.model = model
        self.num_steps = num_steps
        self.step_size = step_size

        self.xs = tf.Variable(np.zeros((1, self.height, self.width, self.channel), dtype=np.float32),
                                    name='modifier')
        self.orig_xs = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])

        self.ys = tf.placeholder(tf.int32, [None])

        y_one_hot = tf.one_hot(self.ys, 10)

        self.epsilon = epsilon

        delta = tf.clip_by_value(self.xs, 0, 1) - self.orig_xs
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs = tf.assign(self.xs, self.orig_xs+delta)

        self.logits = logits = model(self.xs)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_one_hot)
        self.loss = tf.reduce_mean(loss)

        start_vars = set(x.name for x in tf.global_variables())
        self.optimizer = tf.train.AdamOptimizer(step_size*1)

        grad, var = self.optimizer.compute_gradients(self.loss, [self.xs])[0]
        # self.train = self.optimizer.apply_gradients([(tf.sign(grad),var)])
        self.train = self.optimizer.apply_gradients([(tf.sign(grad), var)])

        self.grad = tf.gradients(self.loss, self.xs)[0]

        # gradient palceholder
        self.current_gradient = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        # update using gradient placeholder
        self.update = self.optimizer.apply_gradients([(tf.sign(self.current_gradient), var)])


        end_vars = tf.global_variables()
        self.new_vars = [x for x in end_vars if x.name not in start_vars]

    def perturb(self, x, y, sess):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):

            sess.run(self.train, feed_dict={self.ys: y})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        return sess.run(self.xs)

    def sample_gradient(self, x, y, sess, gradient_samples):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        grad_list = np.zeros((gradient_samples, 1, self.height, self.width, self.channel))
        for grad_idx in range(gradient_samples):
            grad = sess.run(self.grad, feed_dict={self.ys: y})
            grad_list[grad_idx, :, :, :, :] = grad

        return grad_list

    def perturb_gm(self, x, y, sess, gradient_samples):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})


        for i in range(self.num_steps):

            grad_list = np.zeros((gradient_samples, 1, self.height, self.width, self.channel))
            for grad_idx in range(gradient_samples):
                grad = sess.run(self.grad, feed_dict={self.ys: y})
                grad_list[grad_idx, :, :, :, :] = grad
            grad_mean = np.mean(grad_list, axis=0, keepdims=False)


            sess.run(self.update, feed_dict={self.current_gradient: grad_mean})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        return sess.run(self.xs)


# CW-PGD
class CW_PGD(object):
    def __init__(self, model, num_steps, step_size, epsilon, dataset='CIFAR'):
        if dataset == 'CIFAR':
            self.height = 32
            self.width = 32
            self.channel = 3
        elif dataset == 'MNIST':
            self.height = 28
            self.width = 28
            self.channel = 1
        self.model = model
        self.num_steps = num_steps
        self.step_size = step_size

        self.xs = tf.Variable(np.zeros((1, self.height, self.width, self.channel), dtype=np.float32),
                                    name='modifier')
        self.orig_xs = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])

        self.ys = tf.placeholder(tf.int32, [None])

        y_one_hot = tf.one_hot(self.ys, 10)

        self.epsilon = epsilon

        delta = tf.clip_by_value(self.xs, 0, 1) - self.orig_xs
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs = tf.assign(self.xs, self.orig_xs+delta)

        self.logits = logits = model(self.xs)

        label_mask = tf.one_hot(self.ys, 10)

        # code changes here
        target_logit = tf.reduce_sum(label_mask * logits, axis=1)
        other_logit = tf.reduce_max((1-label_mask) * logits - 1e4*label_mask, axis=1)
        self.loss = (other_logit - target_logit)

        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_one_hot)
        # self.loss = tf.reduce_mean(loss)

        start_vars = set(x.name for x in tf.global_variables())
        self.optimizer = tf.train.AdamOptimizer(step_size*1)

        self.grad, var = self.optimizer.compute_gradients(self.loss, [self.xs])[0]
        # self.train = self.optimizer.apply_gradients([(tf.sign(grad),var)])
        self.train = self.optimizer.apply_gradients([(self.grad, var)])

        # self.grad = tf.gradients(self.loss, self.xs)[0]

        # gradient palceholder
        self.current_gradient = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        # update using gradient placeholder
        self.update = self.optimizer.apply_gradients([(self.current_gradient, var)])


        end_vars = tf.global_variables()
        self.new_vars = [x for x in end_vars if x.name not in start_vars]

    def perturb(self, x, y, sess):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):

            sess.run(self.train, feed_dict={self.ys: y})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        return sess.run(self.xs)

    def sample_gradient(self, x, y, sess, gradient_samples):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        grad_list = np.zeros((gradient_samples, 1, self.height, self.width, self.channel))
        for grad_idx in range(gradient_samples):
            grad = sess.run(self.grad, feed_dict={self.ys: y})
            grad_list[grad_idx, :, :, :, :] = grad

        return grad_list

    def perturb_gm(self, x, y, sess, gradient_samples):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):

            grad_list = np.zeros((gradient_samples, 1, self.height, self.width, self.channel))
            for grad_idx in range(gradient_samples):
                grad = sess.run(self.grad, feed_dict={self.ys: y})
                grad_list[grad_idx, :, :, :, :] = grad
            grad_mean = np.mean(grad_list, axis=0, keepdims=False)


            sess.run(self.update, feed_dict={self.current_gradient: grad_mean})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        return sess.run(self.xs)






























