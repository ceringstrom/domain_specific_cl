import tensorflow as tf
import numpy as np

class lossObj:

    #define possible loss functions like dice score, cross entropy, weighted cross entropy

    def __init__(self):
        print('loss init')

    def heavyside_approx(self, x, coef=1, shift=0):
        return 1 / (1 + tf.exp(-coef * (x-shift)))

    def speckle_dice_loss(self, logits, labels, imgs, label_dists, bt_size, alpha=0.01, epsilon=1e-10):
        dice_loss = self.dice_loss_with_backgrnd(logits, labels, epsilon=epsilon)
        speckle_loss, debug = self.speckle_loss(logits, imgs, label_dists, bt_size)
        # ce = self.pixel_wise_cross_entropy_loss(logits, labels)
        # return alpha * dice_loss + (1-alpha) * speckle_loss
        debug = []
        debug.append(alpha * speckle_loss)
        debug.append((1-alpha) * dice_loss)
        # return dice_loss, debug
        return (1-alpha) * dice_loss + alpha * speckle_loss, debug

    def speckle_loss(self, logits, imgs, label_dists, bt_size):
        loss = 0.0
        other_loss = 0.0
        predictions = tf.nn.softmax(logits)
        label_dist_scaling = tf.reduce_mean(label_dists, axis=0)
        with tf.name_scope('speckle_loss'):
            for i in range(bt_size):
                prediction = predictions[i]
                img = imgs[i]
                label_dist = label_dists[i]
                prediction_masks = self.heavyside_approx(prediction, coef=20, shift=0.8)
                # prediction_masks = tf.nn.relu(prediction - 0.5) * 2

                for j in range(1, 4): # replace with number of classes at some point
                    debug = []
                    prediction_mask = tf.expand_dims(prediction_masks[:,:,j], -1)
                    pixels = prediction_mask * img
                    all_ex2 = tf.math.square(pixels)
                    all_ex4 = tf.math.square(all_ex2)
                    approx_num_pix = tf.reduce_sum(prediction_mask)
                    e_x2 = tf.reduce_sum(all_ex2) / approx_num_pix
                    e_x4 = tf.reduce_sum(all_ex4) / approx_num_pix
                    nak_scale = e_x2
                    
                    if(( e_x4 - (e_x2**2)) == 0):
                        nak_shape = 0.0
                    else:
                        nak_shape = e_x2**2 / ( e_x4 - (e_x2**2))
                    nak_scale = tf.expand_dims(nak_scale, 0)
                    nak_shape = tf.expand_dims(nak_shape, 0)
                    # debug.append(nak_shape)
                    # debug.append(nak_scale)
                    # debug.append(tf.concat([nak_shape, nak_scale], axis=0) )
                    # debug.append(label_dist[j-1])
                    # debug.append(label_dist_scaling[j-1])

                    # debug.append((tf.concat([nak_shape, nak_scale], axis=0) - label_dist[j-1]) / label_dist_scaling[j-1])
                    # debug.append(label_dist)
                    new_loss = tf.nn.l2_loss((tf.concat([nak_shape, nak_scale], axis=0) - label_dist[j-1]) / label_dist_scaling[j-1])
                    # debug.append(new_loss)
                    loss = loss + new_loss
        return loss / bt_size, debug
                    

    def dice_loss_with_backgrnd(self, logits, labels, epsilon=1e-10):
        '''
        Calculate a dice loss defined as `1-foreground_dice`. Default mode assumes that the 0 label
         denotes background and the remaining labels are foreground.
        input params:
            logits: Network output before softmax
            labels: ground truth label masks
            epsilon: A small constant to avoid division by 0
        returns:
            loss: Dice loss with background
        '''
        # print(logits.shape)
        # print(labels.shape)
        with tf.name_scope('dice_loss'):

            prediction = tf.nn.softmax(logits)

            intersection = tf.multiply(prediction, labels)
            intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

            l = tf.reduce_sum(prediction, axis=[1, 2])
            r = tf.reduce_sum(labels, axis=[1, 2])

            dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

            loss = 1 - tf.reduce_mean(dices_per_subj)
        return loss

    def dice_loss_without_backgrnd(self, logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
        '''
        Calculate a dice loss of only foreground labels without considering background class.
        Here, label 0 is background and the remaining labels are foreground.
        input params:
            logits: Network output before softmax
            labels: ground truth label masks
            epsilon: A small constant to avoid division by 0
            from_label: First label to evaluate
            to_label: Last label to evaluate
        returns:
            loss: Dice loss without background
        '''
        print(logits.shape)
        print(labels.shape)
        with tf.name_scope('dice_loss'):

            prediction = tf.nn.softmax(logits)

            intersection = tf.multiply(prediction, labels)
            intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

            l = tf.reduce_sum(prediction, axis=[1, 2])
            r = tf.reduce_sum(labels, axis=[1, 2])

            dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

            loss = 1 - tf.reduce_mean(tf.slice(dices_per_subj, (0, from_label), (-1, to_label)))
        return loss

    def pixel_wise_cross_entropy_loss(self, logits, labels):
        '''
        Simple wrapper for the normal tensorflow cross entropy loss
        input params:
            logits: Network output before softmax
            labels: Ground truth masks
        returns:
            loss:  weighted cross entropy loss
        '''
        print(logits.shape)
        print(labels.shape)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return loss

    def pixel_wise_cross_entropy_loss_weighted(self, logits, labels, class_weights):
        '''
        Weighted cross entropy loss, with a weight per class
        input params:
            logits: Network output before softmax
            labels: Ground truth masks
            class_weights: A list of the weights for each class
        returns:
            loss:  weighted cross entropy loss
        '''
        print(logits.shape)
        print(labels.shape)
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * labels, axis=3)

        # For weighted error
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)

        return loss

