import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function

    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    # List all the tensor names to be loaded
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # List comprehension to load all the listed tensors
    vgg_tensor_names = [vgg_input_tensor_name, vgg_keep_prob_tensor_name, vgg_layer3_out_tensor_name,
                        vgg_layer4_out_tensor_name, vgg_layer7_out_tensor_name]
    vgg_tensors = tuple([graph.get_tensor_by_name(tensor_name) for tensor_name in vgg_tensor_names])
    return vgg_tensors

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # Stop backprop to freeze weights
    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)

    with tf.name_scope("decoder"):
        
        '''
        # Scaling
        vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001, name='new_pool3_out_scaled')
        vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01, name='new_pool4_out_scaled')
        '''

        # Upsampling Layer 7 x2
        upsampled_vgg_layer7 = tf.layers.conv2d_transpose(vgg_layer7_out, 2, (3,3), (2,2),
                                                                padding='SAME', name="trn_1")
        # Match the depth of Layer 4 pooling out
        conv1_layer4_pool = tf.layers.conv2d(vgg_layer4_out, 2, (1,1), (1,1), 
                                                padding='SAME', name="trn_2")

        # Skip Layer, 4 -> 7
        skip_layer1 = tf.add(upsampled_vgg_layer7, conv1_layer4_pool, name="trn_3")

        # Upsample the Skip Layer x2
        upsampled_skip_layer1 = tf.layers.conv2d_transpose(skip_layer1, 2, (3,3), (2,2),
                                    padding='SAME', name="trn_4")
        
        # Match the depth of the Layer 3 pooling out
        conv2_layer3_pool = tf.layers.conv2d(vgg_layer3_out, 2, (1,1), (1,1),
                                    padding='SAME', name="trn_5")
        
        # Upsample x8 to match the input dimensions
        final_layer = tf.layers.conv2d_transpose(conv2_layer3_pool, 2, (13, 13), (8, 8),
                                                    padding='SAME', name="trn_6")

    return final_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # Loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = nn_last_layer, labels=correct_label)
    loss_op = tf.reduce_mean(cross_entropy)

    # Accuracy

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Training
    trainable_vars = []

    for variable in tf.trainable_variables():
        print(variable.name)
        if "trn" in variable.name or 'beta' in variable.name:
            trainable_vars.append(variable)

    training_op = optimizer.minimize(loss_op, var_list=trainable_vars)

    return nn_last_layer, training_op, loss_op


    return None, None, None
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
