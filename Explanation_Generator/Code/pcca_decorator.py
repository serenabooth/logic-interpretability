# -*- coding: utf-8 -*-
import time
import copy
from functools import wraps
from pcca_state import *
  

#####################
        
class _PCCA_Reward_Function(object):
  def __init__(self,reward,label):
    self._reward = reward
    self._label = label
    pass

class PCCA_Reward_Function(object): 
  
  def __init__(self, reward, label):
    self._reward_function = _PCCA_Reward_Function(reward,label)

  def __call__(self, func):
    reward_function = self._reward_function

    @wraps(func)
    def wrapped_func(*args):
      PCCA_Function.main_pcca.store_reward(reward_function._reward, PCCA_Function.main_pcca._timer_offset + time.clock())
      return func(*args)

    self._reward_function = None
    return wrapped_func


#####################

class Discontinuity_Function(object):
  def __init__(self):
    pass
  
  def __call__(self, func):
    @wraps(func)
    def wrapped_func(*args):
      PCCA_Function.main_pcca.add_discontinuity(PCCA_Function.main_pcca._timer_offset + time.clock())
      return func(*args)
    return wrapped_func
    

#####################
       
class Planning_Predicate(object):
  predicate_list = [] # List of tuples: (predicate_function_name, predicate_function_pointer, string template)
  
  def __init__(self, template_positive_string, template_negative_string=None):
    self._positive_template_string = template_positive_string
    if template_negative_string is None: template_negative_string = "not " + template_positive_string
    self._negative_template_string = template_negative_string
 
  def __call__(self, func):
    print "Added predicate function %s with template %s" % (func.__name__, self._positive_template_string)
    if func.func_code.co_argcount > 1: print "WARNING: Predicate function %s has more than one argument..." % func.__name__
    Planning_Predicate.predicate_list.append((func.__name__, func, (self._positive_template_string, self._negative_template_string)))
    return func
  
  @staticmethod
  def get_predicate_functions():
    funcs = [p[1] for p in Planning_Predicate.predicate_list]
    return funcs


#####################


class _PCCA_Function(object):
  def __init__(self, nlp_name, preserve_arguments, preserve_return_value):
    self._name = None  # Function name (derived from code)
    self._description = None  # Docstring from function (derived from code)
    self._func = None  # Actual function (pointer to function code)
    self._nlp_name = nlp_name  # Natural language explanation of function
    self._preserve_return_value = preserve_return_value # Save function output with call
    self._preserve_arguments = preserve_arguments # Save function args with call

class PCCA_Function(object):   
  main_pcca = None  # Gets set by the PCCA main class  
  
  def __init__(self, nlp_name=None, preserve_args=False, preserve_return_value=False):
    self._pcca_action = _PCCA_Function(nlp_name,preserve_args,preserve_return_value)

  def __call__(self, func):
    func_name = func.__name__
    if func_name in PCCA_Function.main_pcca._actions:
      # ERROR Condition -- function should only be entered here once
      print 'ERROR - function %s entered into action library more than once?' % (func_name)
      
    self._pcca_action._name = func_name
    self._pcca_action._description = func.__doc__ or ""
    self._pcca_action._func = func

    PCCA_Function.main_pcca._actions[self._pcca_action._name] = self._pcca_action
    func_name = self._pcca_action._name
    preserve_return_value = self._pcca_action._preserve_return_value    
    preserve_arguments = self._pcca_action._preserve_arguments
    
    @wraps(func)
    def wrapped_func(*args):
      #pcca_action = PCCA_Function.main_pcca._actions[func_name]   
      timestamp = PCCA_Function.main_pcca._timer_offset + time.clock()
      before_state = StateVarRef.get_state_snapshot()
      PCCA_Function.main_pcca.add_state_observation(timestamp, before_state)
      str_args = ''
      if preserve_arguments is True:
        str_args = [str(arg) for arg in args]
      trajectory_entry = PCCA_Function.main_pcca.add_trajectory_entry(before_state, func_name, str_args, timestamp)
      output = func(*args)  # Execute actual function
      trajectory_entry._output = None

      if preserve_return_value is True:
        trajectory_entry._output = copy.copy(output)
    
      trajectory_entry._after_timestamp = PCCA_Function.main_pcca._timer_offset + time.clock()
      after_state = StateVarRef.get_state_snapshot()
      trajectory_entry._after_state = after_state      

      PCCA_Function.main_pcca.add_state_observation(trajectory_entry._after_timestamp, after_state)
      return output

    wrapped_func._orig_name = self._pcca_action._name
    self._pcca_action = None # Remove function pointer to allow for easier pickling
    return wrapped_func
    
  @staticmethod
  def from_string(pcca_function_objs_list, action_string):
    '''
    Takes an action string (e.g.: 'move_north(Agent)=None') and returns the object from pcca_function_objs_list that matches it. Returns None if no match found
    '''
    # Deconstruct action_string into function_name, arguments
    action_function = ''
    action_retval = ''
    if '(' in action_string: arg_start_pos = action_string.index('(')
    else: arg_start_pos = len(action_string)
    if '=' in action_string: retval_start_pos = action_string.index('=')
    else: retval_start_pos = 0
    
    action_function = action_string[:arg_start_pos]
    action_retval = action_string[retval_start_pos+1:]
    action_args = action_string[arg_start_pos+1:retval_start_pos-1].split(',')
    for action_obj in pcca_function_objs_list:
      if action_function == action_obj._name: return action_obj
    return None