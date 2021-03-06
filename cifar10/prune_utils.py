import tensorflow as tf
import numpy as np

def get_prune_op(sess, train_gradient, percentage=0.5):
    all_variables = sess.graph.get_collection('variables')
    # ================================================================ #
    # YOUR CODE HERE:
    #   1.find prunable variables i.e. kernel weight/bias
    #   2.prune parameters based on your threshold, calculated based on input argument percentage!
    #   3.get pruned gradient update operator accordingly, save to prune_gradient
    #   example pseudo code for step 2-3:
    #       for var in prunable_variables:
    #           var_np = var.eval()
    #           # construct pruning mask
    #           new_var_np = mask * var_np
    #           prune_op = var.assign(new_var_np)
    #           # apply parameter pruning by sess.run(prune_op)
    #           prune_gradient = train_gradient * mask for each parameter
    #               Example:
    #                   mask = np.zeros(96, dtype=np.float32)
    #                   mask_tensor = tf.convert_to_tensor(mask)
    #                   prune_gradient = [[tf.multiply(v[0], mask_tensor), v[1]] for v in train_gradient if 'fcn/conv1-1/bias' in v[1].name]
    #        return prune_gradient   
    # ================================================================ #
    
    # modify here, no change on gradient for now
    prune_gradient = train_gradient
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    return prune_gradient
