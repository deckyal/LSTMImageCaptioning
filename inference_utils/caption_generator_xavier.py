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
"""Class for generating captions from an image-to-text model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq
import math


import numpy as np


class Caption(object):
  """Represents a complete or partial caption."""

  def __init__(self, sentence, state, logprob, score, metadata=None, group=0):
    """Initializes the Caption.

    Args:
      sentence: List of word ids in the caption.
      state: Model state after generating the previous word.
      logprob: Log-probability of the caption.
      score: Score of the caption.
      metadata: Optional metadata associated with the partial sentence. If not
        None, a list of strings with the same length as 'sentence'.
      group: number associated to the diverse group, group 0 means standard group which refers to traditional Beam Search
    """
    self.sentence = sentence
    self.state = state
    self.logprob = logprob
    self.score = score
    self.metadata = metadata
    self.group = group

  def __cmp__(self, other):
    """Compares Captions by score."""
    assert isinstance(other, Caption)
    if self.score == other.score:
      return 0
    elif self.score < other.score:
      return -1
    else:
      return 1
  
  # For Python 3 compatibility (__cmp__ is deprecated).
  def __lt__(self, other):
    assert isinstance(other, Caption)
    return self.score < other.score
  
  # Also for Python 3 compatibility.
  def __eq__(self, other):
    assert isinstance(other, Caption)
    return self.score == other.score


class TopN(object):
  """Maintains the top n elements of an incrementally provided set."""

  def __init__(self, n):
    self._n = n
    self._data = []

  def size(self):
    assert self._data is not None
    return len(self._data)

  def push(self, x):
    """Pushes a new element."""
    assert self._data is not None
    if len(self._data) < self._n:
      heapq.heappush(self._data, x)
    else:
      heapq.heappushpop(self._data, x)

  def extract(self, sort=False, reset=True):
    """Extracts all elements from the TopN. This is a destructive operation.

    The only method that can be called immediately after extract() is reset().

    Args:
      sort: Whether to return the elements in descending sorted order.

    Returns:
      A list of data; the top n elements provided to the set.
    """
    assert self._data is not None
    data = self._data
    if reset:
      self._data = None
    if sort:
      data.sort(reverse=True)
    return data

  def reset(self):
    """Returns the TopN to an empty state."""
    self._data = []


class CaptionGenerator(object):
  """Class to generate captions from an image-to-text model."""

  def __init__(self,
               model,
               vocab,
               beam_size=3,
               max_caption_length=20,
               length_normalization_factor=0.0):
    """Initializes the generator.

    Args:
      model: Object encapsulating a trained image-to-text model. Must have
        methods feed_image() and inference_step(). For example, an instance of
        InferenceWrapperBase.
      vocab: A Vocabulary object.
      beam_size: Beam size to use when generating captions.
      max_caption_length: The maximum caption length before stopping the search.
      length_normalization_factor: If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.
    """
    self.vocab = vocab
    self.model = model

    self.beam_size = beam_size
    self.max_caption_length = max_caption_length
    self.length_normalization_factor = length_normalization_factor
    #Diverse Beam Search params (note that self.beam_size is the number of beams per group)
    self.G = 3                     # number of diverse groups
    self.B = self.beam_size * self.G    # number of total beams
    self.div_lambda = 0.2          # lambda param controlling level of diversity
    
  def hamming_diversity(self, words_and_probs_cp, past_words):
    """Returns a a probability distribution function of the words with diversity applied"""
    
    n_step_words = len(past_words)
    for w1, p1 in words_and_probs_cp:
      if w1 in past_words:
        words_and_probs_cp[w1] = ( w1, p1+((n_step_words-1)*self.div_lambda) )
      else:
        words_and_probs_cp[w1] = ( w1, p1+((n_step_words)*self.div_lambda) )
          
    return words_and_probs_cp
      
  def top_probable_words(self, words_and_probs_cp, words_and_probs):
    """Returns the top N most probable words with their original probabilities"""
    
    words_and_probs_cp.sort(key=lambda x: -x[1])
    words_and_probs_cp = words_and_probs_cp[0:self.beam_size]
    
    original_top_prob = []
    for i in range(self.beam_size):
      for w, p in words_and_probs:
        if words_and_probs_cp[i][0] == w:
            original_top_prob.append(p)
            
    for i in range(self.beam_size):
      words_and_probs[i] = ( words_and_probs_cp[i][0], original_top_prob[i] )
    words_and_probs = words_and_probs[0:self.beam_size]
          
    return words_and_probs
 
  def top_final_captions(self, partial_captions, complete_captions):
    """Returns the top N most probable final captions"""
    
    total_complete_captions = TopN(self.B)
    
    for g in range(len(complete_captions)):
      complete_captions_list = complete_captions[g].extract()
      for complete_caption in complete_captions_list:
          total_complete_captions.push(complete_caption)

    if total_complete_captions.size():
        return total_complete_captions.extract(sort=True)[0:self.beam_size]
    else:
        return partial_captions[0].extract(sort=True)

  def beam_search(self, sess, encoded_image): 
    #Need to be modified
    """Runs beam search caption generation on a single image.

    Args:
      sess: TensorFlow Session object.
      encoded_image: An encoded image string.

    Returns:
      A list of Caption sorted by descending score.
    """
    # Feed in the image to get the initial state.
    initial_state = self.model.feed_image(sess, encoded_image)

    initial_beam = Caption(
        sentence=[self.vocab.start_id],
        state=initial_state[0],
        logprob=0.0,
        score=0.0,
        metadata=[""])
    
    partial_captions = []
    complete_captions = []
    for i in range(self.G):
        partial_captions.append(TopN(self.beam_size))
        partial_captions[i].push(initial_beam)
        complete_captions.append(TopN(self.beam_size))
                              
    # Run beam search.
    """ for r = 1,... T do """
    for t in range(self.max_caption_length - 1):
      previous_groups = []
      past_words = []
      
      for g in range(0, self.G):
        
        partial_captions_list = partial_captions[g].extract()
        partial_captions[g].reset()
        input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
        state_feed = np.array([c.state for c in partial_captions_list])

        softmax, new_states, metadata = self.model.inference_step(sess,
                                                                input_feed,
                                                                state_feed)

        for i, partial_caption in enumerate(partial_captions_list):
          word_probabilities = softmax[i]
          state = new_states[i]
          # For this partial caption, get the beam_size most probable next words.
          words_and_probs = list(enumerate(word_probabilities))
          words_and_probs_cp = list(enumerate(word_probabilities))
          '''Excepting the first group, apply a diversity penalty to the probability function of the vocabulary'''
          if g != 0:
            pt_cpt_list = partial_captions[g-1].extract(sort=False, reset=False)
            for pt_cpt in pt_cpt_list:
              previous_groups.append(pt_cpt.sentence)
            for pg in previous_groups:
              if t < len(pg):
                if pg[t+1] != self.vocab.end_id:
                  past_words.append(pg[t+1])
            words_and_probs_cp = self.hamming_diversity(words_and_probs_cp, past_words)
            words_and_probs = self.top_probable_words(words_and_probs_cp, words_and_probs)
          else:
            words_and_probs.sort(key=lambda x: -x[1])
            words_and_probs = words_and_probs[0:self.beam_size]
            #print(words_and_probs)
          
          # Each next word gives a new partial caption.
          for w, p in words_and_probs:
            if p < 1e-12:
              continue  # Avoid log(0).
            sentence = partial_caption.sentence + [w]
            logprob = partial_caption.logprob + math.log(p)
            score = logprob       
            if metadata:
              metadata_list = partial_caption.metadata + [metadata[i]]
            else:
              metadata_list = None
            if w == self.vocab.end_id:
              if self.length_normalization_factor > 0:
                score /= len(sentence)**self.length_normalization_factor
              beam = Caption(sentence, state, logprob, score, metadata_list, group=g)
              complete_captions[g].push(beam)
            else:
              beam = Caption(sentence, state, logprob, score, metadata_list, group=g)
              partial_captions[g].push(beam)
        if partial_captions[g].size() == 0:
          # We have run out of partial candidates; happens when beam_size = 1.
          break

    '''# If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_captions[0].size():
      complete_captions[0] = partial_captions[0]'''

    #return complete_captions[1].extract(sort=True)
    return self.top_final_captions(partial_captions, complete_captions)