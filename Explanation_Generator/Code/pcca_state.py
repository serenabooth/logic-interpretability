# -*- coding: utf-8 -*-
import json
import copy
import numpy as np

class State(object):
  def __init__(self, feature_list, predicate_functions):
    self._features = feature_list
    self._predicates = []
    self._predicate_functions = predicate_functions
    
  def __eq__(self, state):
    if state.__class__ == list:
      return self._features == state
    elif state.__class__ == State:
      try:
        return self._features == state._features
      except ValueError:
        if type(self._features) == np.ndarray:
          return np.array_equal(self._features, state._features)
        else:
          print "%s == %s" % (type(self._features), type(state._features))
          return np.all(self._features == state._features)
    else:
      return str(self) == str(state)
  
  def get_features(self):
    return self._features
    
  def get_predicates(self):
    '''
    Returns list of boolean values: one for each predicate function that is applied to this state
    '''
    if self._predicates == []:
      self.compute_predicates(self._predicate_functions)
    return self._predicates, self._predicate_functions
    
  def compute_predicates(self, predicate_functions):
    self._predicates = []
    self._predicate_functions = []
    for func in predicate_functions:
      self._predicates.append(func(self.get_features()))
      self._predicate_functions.append(func)

  def set_predicates(self, predicate_vals, predicate_funcs):
    self._predicates = copy.copy(predicate_vals)
    self._predicate_functions = copy.copy(predicate_funcs)

  def __ne__(self, state):
    return not self.__eq__(state)

  def __str__(self):
    return json.dumps(self._features)

  def __hash__(self):
    return hash(json.dumps(self._features))


     
class StateVarRef:
    '''
    Python doesn't let you make pointers to variables, only to the objects they reference.
    In order to keep track of any world-state variables (or in this case, program state variables), it is necessary 
    '''
    state_vars = []

    @staticmethod
    def get_state_snapshot():
        state = []
        for var in StateVarRef.state_vars:
            state.append(copy.copy(var.get()))
        return state

    def __init__(self,obj): 
        self.obj = obj
        StateVarRef.state_vars.append(self)
    def get(self): return self.obj
    def set(self, obj): self.obj = obj    
    def __str__(self):
      return str(self.obj)




class TrajectoryEntry(object):
  def __init__(self,state,func,args,timestamp=None):
    '''
    @param state - Current world state of function call
    @param args - Array of serialized function arguments
    @param timestamp - System time of function call
    '''
    self._func_name = func
    self._args = args
    self._before_timestamp = timestamp
    self._after_timestamp = timestamp
    self._output = None
    self._before_state = state
    self._after_state = None
     
  def __str__(self):
    if len(self._args) > 0:
        args_str = ','.join(self._args)
    else:
        args_str = ''
    return "%s(%s)=%s" % (self._func_name, args_str, str(self._output))

class RewardEntry(object):
  def __init__(self, reward, timestamp):
    self._reward = reward
    self._timestamp = timestamp
  
  def __str__(self):
    return str(self._reward)
