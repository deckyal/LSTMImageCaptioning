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
"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops


class ShowAndTellModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None


    self.listState = []
    self.listOutput = []
    self.seq_run = 0
                
    self.allExamined = self.config.allExamined;
    self.n = self.config.n
    
    
    self.wa = tf.constant(0,shape=[self.config.num_lstm_units*2],dtype=tf.float32)
    self.wc = tf.constant(0,shape=[self.config.num_lstm_units, self.config.num_lstm_units*4],dtype=tf.float32)

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self): #Note, preprocess 1
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)
      
      
      ######
      # Prefetch serialized SequenceExample protos.
      '''input_queue = input_ops.prefetch_input_data(
          self.reader,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))      '''
      
      ###
      
      
      print(self.images)
      print(self.input_seqs)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))

    self.images = images
    self.input_seqs = input_seqs
    
    self.target_seqs = target_seqs
    self.input_mask = input_mask
    
    
    print(self.images)
    print(self.input_seqs)
    print(self.input_mask)

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """
    inception_output = image_embedding.inception_v3(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

    self.seq_embeddings = seq_embeddings
    
  def build_attention_variable(self):
      
    with tf.variable_scope("attention_var"), tf.device("/cpu:0"):
      wa = tf.get_variable(
          name="wa",
          shape=[self.config.num_lstm_units*2],
          initializer=self.initializer)
      wc = tf.get_variable(
          name="wc",
          shape=[self.config.num_lstm_units, self.config.num_lstm_units*4],
          initializer=self.initializer)

    self.wa = wa
    self.wc = wc
    

  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    if self.mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell.zero_state(
          batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
      _, initial_state = lstm_cell(self.image_embeddings, zero_state)

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()

      if self.mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        
        # I think we canged that function, bring it back to the original one? TODO
        
        tf.concat(axis=1, values=initial_state, name="initial_state")
        
        # Placeholder for the current sequence number 
        seq_number = tf.placeholder(dtype=tf.int32,
                                    shape=[1],
                                    name="seq_number")
        
        # Placeholder for the list of past states H 
        state_listH = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None,None],
                                    name="state_listH")
        
        # Placeholder for the list of past states C
        state_listC = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None,None],
                                    name="state_listC")
        
        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
        
        print(state_tuple)
        state_tuple[0] = tf.Print(state_tuple[0], ["Ei ",tf.shape(state_tuple[0])]) #C
        state_tuple[1] = tf.Print(state_tuple[1], ["Ei ",tf.shape(state_tuple[1])]) #H
        
        the_input = tf.squeeze(self.seq_embeddings, axis=[1])
        
        the_input = tf.Print(the_input, ["the_input ",tf.shape(the_input)])
        
        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            inputs=the_input,
            state=state_tuple)
        
        
        lstm_outputs = tf.Print(lstm_outputs, ["lstm output alter  ",tf.shape(lstm_outputs)])
        #state_tuple = tf.Print(state_tuple, ["State tuple end",tf.shape(state_tuple)],name="yourmom")
        
        lastStateH = tf.squeeze(state_tuple.h,name="lastStateH")
        lastStateC = tf.squeeze(state_tuple.c,name="lastStateC")
        
        seq_number_s = tf.squeeze(seq_number)
        
        print("Beam size ",self.config.beam_size)
        #Now do the attention mechanism
        if self.config.doAttention  : 
            
            def f11(lastOutput):
                return lastOutput
            
            def f22(lastStateH,stateH_list,lastStateC,stateC_list,seq_number_s,lastOutput):
                
                lim = seq_number_s
                attention_vector = []
                
                for beam in range(self.config.beam_size) : #Beam search count
                    
                    contextList = tf.TensorArray(size = 1,dynamic_size = True,dtype=tf.float32,name="CL")
                    
                    score = tf.constant(0,shape=[1],dtype=tf.float32,name="SC")
                    
                    if self.allExamined :  
                        i_bef = tf.constant(0,dtype=tf.int32)
                    else : 
                        i_bef = tf.maximum(0,tf.subtract(seq_number_s,self.n))
                    
                    i_bef = tf.Print(i_bef,["The j ",i_bef])
                    lim = tf.Print(lim,["The lim ",lim])
                    
                    
                    
                    
                    def condition3(i_bef,stateH_list,lastStateH,stateC_list,lastStateC,score,contextList,lim):
                        i_bef = tf.Print(i_bef,["in condition 2",i_bef])
                        
                        return tf.less(i_bef,lim)
                    
                    def body3(i_bef,stateH_list,lastStateH,stateC_list,lastStateC,score,contextList,lim):
                        i_bef = tf.Print(i_bef,["in body 2 ",i_bef])
                        curState = tf.concat([lastStateC[beam],lastStateH[beam]],axis = 0)
                        prevState = tf.concat([stateC_list[i_bef][beam],stateH_list[i_bef][beam]],axis = 0)
                        
                        
                        prevState = tf.Print(prevState,['prev state ',prevState])
                        curState = tf.Print(curState,['cur state ',curState])
                        
                        c_score = tf.matmul(
                                    tf.expand_dims(curState,0),
                                        tf.matmul(
                                            tf.diag(self.wa),tf.expand_dims(prevState,1))
                                        )
                        
                        c_score = tf.Print(c_score,['c score ',c_score])
                        
                        imScore = tf.exp(c_score)
                        
                        curContext = tf.squeeze(imScore) * prevState
                        
                        curContext = tf.Print(curContext,['cur context',curContext])
                        
                        contextList = contextList.write(i_bef,curContext)
                        
                        imScore = tf.Print(imScore,["imscore ",imScore])
                        
                        score += tf.squeeze(imScore)
                        score.set_shape([1])
                        
                        score = tf.Print(score,["Score : ",score])
                        return [tf.add(i_bef,1),stateH_list,lastStateH,stateC_list,lastStateC,score,contextList,lim]
                    
                    res = tf.while_loop(condition3,body3,loop_vars = (i_bef,stateH_list,lastStateH,stateC_list,lastStateC,score,contextList,lim))
                    
                    
                    i_bef = tf.Print(i_bef,["The j222 ",i_bef])
                    lim = tf.Print(lim,["The lim222 ",lim])
                    
                    
                    score = res[5]
                        
                    score = tf.Print(score,["Scoreall : ",score])
                    
                    contextList  = res[6]
                    
                    '''contextList.write(0,tf.zeros([3]))
                    contextList.write(1,tf.zeros([3]))
                    contextList.write(2,tf.zeros([3]))
                    contextList.write(3,tf.zeros([3]))'''
                    contextListed = contextList.stack()
                    
                    #### end loop of score
                    #contextList = tf.Print(contextList,["CL",contextList])
                    
                    score = tf.Print(score,["Scoreall : ",score])
                    
                    contextListed = tf.Print(contextListed,["contextListed Bef 1: ",contextListed,tf.shape(contextListed)])
                    
                    contextListed = tf.divide(contextListed,score) #normalize by all score
                    contextListed = tf.Print(contextListed,["contextListed all 2: ",contextListed,tf.shape(contextListed)])
                    
                    contextListed = tf.reduce_sum(contextListed,axis=0) #sum all 
                    contextListed = tf.Print(contextListed,["contextListed : divide 3",contextListed,tf.shape(contextListed)])
                    
                    
                    print("the shape ",contextListed.shape)
                    
                    contextListed = tf.expand_dims(contextListed,1)
                    lsh = tf.expand_dims(tf.concat([lastStateC[beam],lastStateH[beam]],axis = 0),1)
                    #### calcualte attention vector 
                    the_concat = tf.concat([contextListed,lsh],axis = 0)
                    
                    print("concat shape",the_concat.shape)
                    attention_vector.append(
                        tf.squeeze(
                            tf.tanh(
                                tf.matmul(
                                    self.wc,the_concat))))
                    #### this is ht end of loop of context
                    
                return tf.stack(attention_vector)
                
            lstm_outputs = tf.cond(tf.less(seq_number_s, 1), lambda:f11(lstm_outputs), lambda:f22(lastStateH,state_listH,lastStateC,state_listC,seq_number_s,lstm_outputs))
                
        # Concatentate the resulting state.
        tf.concat(axis=1, values=state_tuple, name="state")
      else:
        # Run the batch of sequence embeddings through the LSTM.
        sequence_length = tf.reduce_sum(self.input_mask, 1)
        
        print(sequence_length)
        
        # Here we will need to call the self.attention_w variable that we created TODO
        
        # We will need to update the attention lists, I think that state_list and output_list need to be replaced with the previous self var TODO
        
        '''for i in range(0, sequence_length): 
            if i==0:
                merged = self.seq_embeddings
                LSTMState = initial_state
            else: 
                merged = lstm_outputs
                
            lstm_outputs, LSTMState= lstm_cell(merged, LSTMState) #lstm_outputs will be a sequence of logits? 1 logits for each generated word?
            
            if i > 0:
                score = 0
                alpha_score = []
                
                for j in range(0, i):
                    score += tf.dot(lstm_outputs,tf.matmul(wa,output_list[j]))
                    
                alpha_score.append(tf.exp(tf.dot(lstm_outputs,tf.matmul(wa,output_list[i-1])))/tf.exp(score))
                
                context = 0
                
                for j in range(0, i):
                    context += alphaa_score[j]*output_list[j]
                    
                attention_vector = tf.tanh(tf.matmul(wc,tf.concat([context,lstm_outputs],axis=2)))
                
                lstm_outputs = attention_vector
                
            state_list.append(LSTMState)
            output_list.append(lstm_outputs)'''
        
        print(self.seq_embeddings)
        print(initial_state)    
        print(sequence_length)
        
        sequence_length = tf.Print(sequence_length,["Seq length : ",sequence_length])
        
        #first try to convert the normal one : 
        
        
        self.seq_embeddings = tf.Print(self.seq_embeddings, ["Seq embedding ",tf.shape(self.seq_embeddings)])
        
        '''lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=self.seq_embeddings,
                                            sequence_length=sequence_length,
                                            initial_state=initial_state,
                                            dtype=tf.float32,
                                            scope=lstm_scope)
        
        lstm_outputs = tf.Print(lstm_outputs, ["lstm output original  ",tf.shape(lstm_outputs)])'''
        
        
        '''i = tf.constant(0,dtype=tf.int32)
        max_length = tf.reduce_max(sequence_length)
        
        print(max_length,'testing')
        
        
        lastStateC = initial_state.c
        lastStateH = initial_state.h
        lastOutput = tf.constant(0,shape=[32,512],dtype=tf.float32)
        stateC_list = tf.TensorArray(size = 1,dynamic_size = True,dtype=tf.float32)
        stateH_list = tf.TensorArray(size = 1,dynamic_size = True,dtype=tf.float32)
        output_list = tf.TensorArray(size = 1,dynamic_size = True,dtype=tf.float32)
        
        def condition(i,stateC_list,stateH_list,output_list,lastStateC,lastStateH,lastOutput):
            return tf.less(i,max_length)
                
        def body(i,stateC_list,stateH_list,output_list,lastStateC,lastStateH,lastOutput):
            
            lastState = tf.nn.rnn_cell.LSTMStateTuple(lastStateC, lastStateH)
            
            def f1():
                return self.seq_embeddings[:,i]
            def f2(): 
                #return lastOutput
                return self.seq_embeddings[:,i]
            
            inputI = tf.cond(tf.less(i, 1), f1, f2)
            
            lastOutput, lastState= lstm_cell(inputI, lastState) #lstm_outputs will be a sequence of logits? 1 logits for each generated word?
            
            lastStateC = lastState.c
            lastStateH = lastState.h
            
            stateC_list=stateC_list.write(i,lastState.c)
            stateH_list=stateH_list.write(i,lastState.h)
            output_list=output_list.write(i,lastOutput)
            
            return [tf.add(i,1),stateC_list,stateH_list,output_list,lastStateC,lastStateH,lastOutput] 
        
        
        lstm_outputs = tf.while_loop(
            condition,body,
            loop_vars = (i,stateC_list,stateH_list,output_list,lastStateC,lastStateH,lastOutput),
            #lastout
            )
        
        lstm_outputs = lstm_outputs[3].stack()
        lstm_outputs = tf.transpose(lstm_outputs, [1,0,2], name = "result_unwrapped_stats")'''
          
        i = tf.constant(0,dtype=tf.int32)
        max_length = tf.reduce_max(sequence_length)
        
        print(max_length,'testing')
        
        
        lastStateC = initial_state.c
        lastStateH = initial_state.h
        lastOutput = tf.constant(0,shape=[self.config.batch_size,512],dtype=tf.float32)
        stateC_list = tf.TensorArray(size = 1,dynamic_size = True,dtype=tf.float32,name="state_c",clear_after_read=False)
        stateH_list = tf.TensorArray(size = 1,dynamic_size = True,dtype=tf.float32,name="state_h",clear_after_read=False)
        output_list = tf.TensorArray(size = 1,dynamic_size = True,dtype=tf.float32,name="output")
        
        def condition(i,stateC_list,stateH_list,output_list,lastStateC,lastStateH,lastOutput):
            return tf.less(i,max_length)
                
        def body(i,stateC_list,stateH_list,output_list,lastStateC,lastStateH,lastOutput):
            
            lastState = tf.nn.rnn_cell.LSTMStateTuple(lastStateC, lastStateH)
            
            inputI = self.seq_embeddings[:,i]
            
            lastOutput, lastState= lstm_cell(inputI, lastState) #lstm_outputs will be a sequence of logits? 1 logits for each generated word?
            
            stateC_list=stateC_list.write(i,lastState.c)
            stateH_list=stateH_list.write(i,lastState.h)
            lastStateC = lastState.c
            lastStateH = lastState.h
            
            def f11(lastOutput):
                lastOutput = tf.Print(lastOutput, ["Iteration 0 ",lastOutput])
                return lastOutput
            
            def f22(lastStateH,stateH_list):
                #return lastOutput
                
                ###this is loop of att weiights, and context
                
                lim = i#tf.subtract(i,1)
                attention_vector = []
                
                for z in range(self.config.batch_size):
                    
                    contextList = tf.TensorArray(size = 1,dynamic_size = True,dtype=tf.float32,name="CL")
                    score = tf.constant(0,shape=[1],dtype=tf.float32,name="SC")
                    
                    if self.allExamined :  
                        j = tf.constant(0,dtype=tf.int32)
                    else : 
                        j = tf.maximum(0,tf.subtract(i,self.n))
                            
                    def condition2(j,stateH_list,lastStateH,stateC_list,lastStateC,score,contextList):
                        return tf.less(j,lim)
                    
                    def body2(j,stateH_list,lastStateH,stateC_list,lastStateC,score,contextList):
                        print(tf.expand_dims(lastStateH[z],0).shape,self.wa.shape,stateH_list.read(j)[z].shape)
                        
                        curState = tf.concat([lastStateC[z],lastStateH[z]],axis = 0)
                        prevState = tf.concat([stateC_list.read(j)[z],stateH_list.read(j)[z]],axis = 0)
                        
                        imScore = tf.exp(
                            tf.matmul(
                                    tf.expand_dims(curState,0),
                                        tf.matmul(
                                            tf.diag(self.wa),tf.expand_dims(prevState,1))
                                        )
                                    )
                        
                        print(tf.squeeze(imScore).shape)
                        
                        curContext = tf.squeeze(imScore) * prevState
                        contextList = contextList.write(j,curContext)
                        
                        score += tf.squeeze(imScore)
                        score.set_shape([1])
                        return [tf.add(j,1),stateH_list,lastStateH,stateC_list,lastStateC,score,contextList]
                    
                    score = tf.while_loop(condition2,body2,loop_vars = (j,stateH_list,lastStateH,stateC_list,lastStateC,score,contextList))[5]
                    
                    #### end loop of score
                    
                    contextList = contextList.stack()
                    contextList = tf.divide(contextList,score) #normalize by all score
                    contextList = tf.reduce_sum(contextList,axis=0) #sum all 
                    
                    print("the shape ",contextList.shape)
                    
                    contextList = tf.expand_dims(contextList,1)
                    lsh = tf.expand_dims(tf.concat([lastStateC[z],lastStateH[z]],axis = 0),1)
                    #### calcualte attention vector 
                    the_concat = tf.concat([contextList,lsh],axis = 0)
                    
                    print("concat shape",the_concat.shape)
                    attention_vector.append(
                        tf.squeeze(
                            tf.tanh(
                                tf.matmul(
                                    self.wc,the_concat))))
                    #### this is ht end of loop of context
                    
                return tf.stack(attention_vector)
                
            lastOutput = tf.cond(tf.less(i, 1), lambda:f11(lastOutput), lambda:f22(lastStateH,stateH_list))
            
            lastOutput.set_shape([self.config.batch_size,512])
            
            output_list=output_list.write(i,lastOutput)
            
            return [tf.add(i,1),stateC_list,stateH_list,output_list,lastStateC,lastStateH,lastOutput] 
        
        
        lstm_outputs = tf.while_loop(
            condition,body,
            loop_vars = (i,stateC_list,stateH_list,output_list,lastStateC,lastStateH,lastOutput),
            #lastout
            )
        
        lstm_outputs = lstm_outputs[3].stack()
        lstm_outputs = tf.transpose(lstm_outputs, [1,0,2], name = "result_unwrapped_stats")
    
        print("output",lstm_outputs)
        lstm_outputs = tf.Print(lstm_outputs, ["lstm output alter  ",tf.shape(lstm_outputs)])
        
    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])
    
    #modify the inference result 

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)

    # TODO do text simplification

    if self.mode == "inference":
      tf.nn.softmax(logits, name="softmax")
    else:
      targets = tf.reshape(self.target_seqs, [-1])
      weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

      # Compute losses.
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                              logits=logits)
      batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                          tf.reduce_sum(weights),
                          name="batch_loss")
      tf.losses.add_loss(batch_loss)
      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss
      self.target_cross_entropy_losses = losses  # Used in evaluation.
      self.target_cross_entropy_loss_weights = weights  # Used in evaluation.


  def initialize_uninitialized_global_variables(self,sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars)) 


  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
          
        tf.logging.info("Restoring Inception2 variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        
        tf.logging.info("Initializing attention var")
        
        saver.restore(sess, self.config.inception_checkpoint_file)
        
      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.build_attention_variable()

    self.setup_inception_initializer()
    self.setup_global_step()