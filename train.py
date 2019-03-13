# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import configuration
import show_and_tell_model

base_folder = "/media/deckyal/78DCCAA06404648F"

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "",
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 4004300, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 5,
                        "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  '''assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.train_dir, "--train_dir is required"'''

  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = base_folder+"/mscoco/train-?????-of-00256"#FLAGS.input_file_pattern
  model_config.inception_checkpoint_file = base_folder+"/model/inception_v3.ckpt"#FLAGS.inception_checkpoint_file
  training_config = configuration.TrainingConfig()

  # Create training directory.
  train_dir = base_folder + "/model/3m_wa_n_all/train"#FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model = show_and_tell_model.ShowAndTellModel(
        model_config, mode="train", train_inception=FLAGS.train_inception)
    model.build()

    # Set up the learning rate.
    learning_rate_decay_fn = None
    if FLAGS.train_inception:
      learning_rate = tf.constant(training_config.train_inception_learning_rate)
    else:
      learning_rate = tf.constant(training_config.initial_learning_rate)
      if training_config.learning_rate_decay_factor > 0:
        num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                 model_config.batch_size)
        decay_steps = int(num_batches_per_epoch *
                          training_config.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
          return tf.train.exponential_decay(
              learning_rate,
              global_step,
              decay_steps=decay_steps,
              decay_rate=training_config.learning_rate_decay_factor,
              staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)

    # Set up the Saver for saving and restoring model checkpoints.
        
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)#, scope = the_scope)
    print("VL  ",var_list)
    
    saver = tf.train.Saver(var_list = var_list,max_to_keep=training_config.max_checkpoints_to_keep,save_relative_paths=True)
    var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = '(?!attention_var)')
    print("test",var_list2)
    saver2 = tf.train.Saver(var_list = var_list2)

    attVar = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="attention_var")
    local_init_op = tf.variables_initializer(attVar)
    

  def init_fn_2(sess): 
    tf.logging.info("Restoring the original weight %s",
                    model_config.inception_checkpoint_file)
    checkpoint_path2 = model_config.base_folder+"/model/3m/train/model.ckpt-3003677"
     
    saver2.restore(sess,checkpoint_path2)
    sess.run(local_init_op)    
  
  
  config = tf.ConfigProto()
  config.gpu_options.visible_device_list = "0"
  gpu_fract = .75
  config.gpu_options.per_process_gpu_memory_fraction = gpu_fract
  
  # Run training.
  tf.contrib.slim.learning.train(
      train_op,
      train_dir,
      log_every_n_steps=FLAGS.log_every_n_steps,
      graph=g,
      global_step=model.global_step,
      number_of_steps=FLAGS.number_of_steps,
      init_fn=init_fn_2,
      saver=saver,
      #local_init_op = local_init_op,
      session_config = config)


if __name__ == "__main__":
  tf.app.run()
