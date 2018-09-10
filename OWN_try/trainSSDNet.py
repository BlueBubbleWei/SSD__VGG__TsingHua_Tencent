import tensorflow as tf
import model_deploy
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import tf_utils
slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'

# =========================================================================== #
# SSD Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'loss_alpha', 1., '损失函数中的alpha')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., '损失函数中下降率')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, '损失函数中的匹配阈值')

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/','checkpoints的写入空间')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,'使用CPU部署clones')
tf.app.flags.DEFINE_integer('num_readers', 4,'平行读取数据集线程的数量')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4, '创建batch的线程的数量')

tf.app.flags.DEFINE_integer('log_every_n_steps', 10,'打印的间隔数')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,'保存summaries的间隔，以秒计算')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600, 'T保存model的间隔数，以秒计算')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU fraction的使用')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, '模型权重的下降率')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop','优化器的名称"adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95, 'adadelta的下降率')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,'AdaGrad加速器的开始时间')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,'滑动平均模型的第一次动态加速')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    '滑动平均模型的第二次动态加速')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5, '学习率的开平方')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,'FTRL accumulators 的初始值.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9, ' MomentumOptimizer and RMSPropOptimizer 的动能.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, '动能初始值.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type','exponential',
    '指定学习率是如何下降的. One of "fixed", "exponential",'' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, '初始化学习率')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float('label_smoothing', 0.0, '标签平滑的总量')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '学习率下降参数.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    '使用滑动均值的下降率.如果剩余为空值，滑动均值就不会使用')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_2007', '加载数据集的名称')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, '数据集的类别')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', '根据 train/test划分名称.')
tf.app.flags.DEFINE_string(
    'dataset_dir', '../../dataset/dealVOC', '数据集文件存储的路径。')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_300_vgg', '训练的网络名称.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, '每个batch中的样本数量')
tf.app.flags.DEFINE_integer(
    'train_image_size', 300, '训练图片的尺寸')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,'训练次数的最大值.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,'fine-tune checkpoint的路径.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    '通过设置默认值，空值会训练所有的参数')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,'重新存储 checkpoint忽视缺失值.')

FLAGS = tf.app.flags.FLAGS



def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # Config model_deploy. Keep TF Slim Models structure.
        # Useful if want to need multiple GPUs and/or servers in the future.
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0)
        # Create global_step.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        # Select the dataset.
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=True)

        tf_utils.print_configuration(FLAGS.__flags, ssd_params,
                                     dataset.data_sources, FLAGS.train_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.device(deploy_config.inputs_device()):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size,
                    shuffle=True)
            # Get for SSD network: image, labels, bboxes.
            [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                             'object/label',
                                                             'object/bbox'])
            # Pre-processing image, labels and bboxes.
            image, glabels, gbboxes = \
                image_preprocessing_fn(image, glabels, gbboxes,
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT)
            # Encode groundtruth labels and bboxes.
            gclasses, glocalisations, gscores = \
                ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
            batch_shape = [1] + [len(ssd_anchors)] * 3
    #
            # Training batches and queue.
            r = tf.train.batch(
                tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
            b_image, b_gclasses, b_glocalisations, b_gscores = \
                tf_utils.reshape_list(r, batch_shape)

            # Intermediate queueing: unique batch computation pipeline for all
            # GPUs running the training.
            batch_queue = slim.prefetch_queue.prefetch_queue(
                tf_utils.reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]),
                capacity=2 * deploy_config.num_clones)
    #
    #     # =================================================================== #
    #     # Define the model running on every GPU.
    #     # =================================================================== #
        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple
            clones of network_fn."""
            # Dequeue batch.
            b_image, b_gclasses, b_glocalisations, b_gscores = \
                tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)

            # Construct SSD network.
            arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                          data_format=DATA_FORMAT)
            with slim.arg_scope(arg_scope):
                predictions, localisations, logits, end_points = \
                    ssd_net.net(b_image, is_training=True)
    #         # Add loss function.
            ssd_net.losses(logits, localisations,
                           b_gclasses, b_glocalisations, b_gscores,
                           match_threshold=FLAGS.match_threshold,
                           negative_ratio=FLAGS.negative_ratio,
                           alpha=FLAGS.loss_alpha,
                           label_smoothing=FLAGS.label_smoothing)
            return end_points
    #
    #     # Gather initial summaries.
    #     summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    #
    #     # =================================================================== #
    #     # Add summaries from first clone.
    #     # =================================================================== #
    #     clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    #     first_clone_scope = deploy_config.clone_scope(0)
    #     # Gather update_ops from the first clone. These contain, for example,
    #     # the updates for the batch_norm variables created by network_fn.
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
    #
    #     # Add summaries for end_points.
    #     end_points = clones[0].outputs
    #     for end_point in end_points:
    #         x = end_points[end_point]
    #         summaries.add(tf.summary.histogram('activations/' + end_point, x))
    #         summaries.add(tf.summary.scalar('sparsity/' + end_point,
    #                                         tf.nn.zero_fraction(x)))
    #     # Add summaries for losses and extra losses.
    #     for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
    #         summaries.add(tf.summary.scalar(loss.op.name, loss))
    #     for loss in tf.get_collection('EXTRA_LOSSES', first_clone_scope):
    #         summaries.add(tf.summary.scalar(loss.op.name, loss))
    #
    #     # Add summaries for variables.
    #     for variable in slim.get_model_variables():
    #         summaries.add(tf.summary.histogram(variable.op.name, variable))
    #
    #     # =================================================================== #
    #     # Configure the moving averages.
    #     # =================================================================== #
    #     if FLAGS.moving_average_decay:
    #         moving_average_variables = slim.get_model_variables()
    #         variable_averages = tf.train.ExponentialMovingAverage(
    #             FLAGS.moving_average_decay, global_step)
    #     else:
    #         moving_average_variables, variable_averages = None, None
    #
    #     # =================================================================== #
    #     # Configure the optimization procedure.
    #     # =================================================================== #
    #     with tf.device(deploy_config.optimizer_device()):
    #         learning_rate = tf_utils.configure_learning_rate(FLAGS,
    #                                                          dataset.num_samples,
    #                                                          global_step)
    #         optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
    #         summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    #
    #     if FLAGS.moving_average_decay:
    #         # Update ops executed locally by trainer.
    #         update_ops.append(variable_averages.apply(moving_average_variables))
    #
    #     # Variables to train.
    #     variables_to_train = tf_utils.get_variables_to_train(FLAGS)
    #
    #     # and returns a train_tensor and summary_op
    #     total_loss, clones_gradients = model_deploy.optimize_clones(
    #         clones,
    #         optimizer,
    #         var_list=variables_to_train)
    #     # Add total_loss to summary.
    #     summaries.add(tf.summary.scalar('total_loss', total_loss))
    #
    #     # Create gradient updates.
    #     grad_updates = optimizer.apply_gradients(clones_gradients,
    #                                              global_step=global_step)
    #     update_ops.append(grad_updates)
    #     update_op = tf.group(*update_ops)
    #     train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
    #                                                       name='train_op')
    #
    #     # Add the summaries from the first clone. These contain the summaries
    #     summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
    #                                        first_clone_scope))
    #     # Merge all summaries together.
    #     summary_op = tf.summary.merge(list(summaries), name='summary_op')
    #
    #     # =================================================================== #
    #     # Kicks off the training.
    #     # =================================================================== #
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    #     config = tf.ConfigProto(log_device_placement=False,
    #                             gpu_options=gpu_options)
    #     saver = tf.train.Saver(max_to_keep=5,
    #                            keep_checkpoint_every_n_hours=1.0,
    #                            write_version=2,
    #                            pad_step_number=False)
    #     slim.learning.train(
    #         train_tensor,
    #         logdir=FLAGS.train_dir,
    #         master='',
    #         is_chief=True,
    #         init_fn=tf_utils.get_init_fn(FLAGS),
    #         summary_op=summary_op,
    #         number_of_steps=FLAGS.max_number_of_steps,
    #         log_every_n_steps=FLAGS.log_every_n_steps,
    #         save_summaries_secs=FLAGS.save_summaries_secs,
    #         saver=saver,
    #         save_interval_secs=FLAGS.save_interval_secs,
    #         session_config=config,
    #         sync_optimizer=None)


if __name__ == '__main__':
    tf.app.run()

