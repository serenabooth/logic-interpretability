# -*- coding: utf-8 -*-
import time
import dill as pickle
import json
import re
import networkx as nx
import sys
from networkx.readwrite import json_graph

from bisect import bisect_left
from qm import *
import os.path

from pcca_state import *
from pcca_decorator import *
import ast

import datetime

#BRAD'S TODO:
#  - Add individual reward channels as well as 'global state reward'
#  - Include state visit frequency as feature to track/cluster on for nominal behavior detection / cluster anchoring
#       - Add cluster visualization mode to website
#  - Add slider bar to index.html to progress through each graph
  
time.clock() # Start execution clock

class PCCA(object):

  def __init__(self,checkpoint_interval=250):
    self._program_name = __file__
    self._graph = None
    self._simulator_world = None
    self._planning_predicates = Planning_Predicate.predicate_list # Contains list of (func.__name__, func, (self._positive_template_string, self._negative_template_string)) tuples
    self._state_history = []
    self._state_history_timestamps = []
    self._checkpoint_interval = checkpoint_interval
    self._last_timestamp_graphed = 0
    self._transition_counts = {} # Dict[State str][Action str][State str] = TransitionCount int
    self._timer_offset = time.time() # Accommodate fact that Windows machines use clock() to report delta since its first call, while UNIX
                                     # uses it to report raw CPU clock time
    self._rewards_applied = [] # (reward, timestamp) tuples to apply
    self._discontinuities = [] # timestamp discontinuities to use as boundaries for halting reward propagation
    
    # Load behavior graph from previous examples, if it exists
    graph_filename = self._program_name+".graph"
    if os.path.isfile(graph_filename):
      self._graph = PCCA.load_from_file(graph_filename)
    else:
      self._graph = nx.MultiDiGraph()
      
    self._states = []
    self._actions = {}
    self._call_trajectory = []
    
  @staticmethod
  def load_from_file(filename):
    try:
      f = open(filename+'-graph.pkl', 'rb')
      graph = pickle.load(f)
      f.close()
      f = open(filename+'-pcca.pkl', 'rb')
      model = pickle.load(f)
      f.close()
    except:
      e = sys.exc_info()[0]
      print "Could not open graph file. -- " + str(e)
      raise Exception('Could not open graph file')

    model._graph = graph
    return model

  def save_predicates(self):
    self._planning_predicates = Planning_Predicate.predicate_list
    f = open('predicates.txt', 'wb')

    i = 1
    for predicate in Planning_Predicate.predicate_list:
      f.write(str("idx: " + str(i) + " - "))
      i += 1
      f.write(str(predicate[2][0]))
      f.write("\n")
    f.close()

  def save_to_file(self, filename):  
    self._planning_predicates = Planning_Predicate.predicate_list # Save predicates

    #graph_data = json_graph.node_link_data(self._graph)
    f = open(filename+'-graph.pkl','wb')
    pickle.dump(self._graph,f)
    f.close()
    tmp_g = self._graph
    tmp_a = self._actions
    self._graph = None
    #self._actions = None
    f = open(filename+'-pcca.pkl','wb')
    pickle.dump(self,f)
    f.close()
    self._graph = tmp_g
    self._actions = tmp_a
    return

  def export_graph_json(self,filename=None):
    # Turn graph into JSON, return a JSON obj string if filename is None, otherwise write to filename
    self.compile_to_graph()
    graph_data = json_graph.node_link_data(self._graph)
    graph_data['pcca_info'] = {}
    graph_data['pcca_info']['predicates'] = self._graph.predicates
    json_str = json.dumps(graph_data)
    if filename is None:
      return json_str
    else:
      output_str = 'graph = %s; graph.action_count=%d;' % (json_str, len(self._call_trajectory))
      f = open(filename,'w')
      f.write(output_str)
      f.close()
      return

  def add_state_observation(self, timestamp, state_vector):
    state = State(state_vector,Planning_Predicate.get_predicate_functions())    
    self._state_history.append(state)
    self._state_history_timestamps.append(timestamp)
    print "State history contains %d states -- added: %s" % (len(self._state_history), str(state_vector))
    return state
    
  def get_nearest_state_idx(self, timestamp):
    state_idx = bisect_left(self._state_history_timestamps, timestamp + 1E-5)
    if state_idx == 0: return 0 #None
    return state_idx - 1
    
  @staticmethod
  def binary_search(a, x, lo=0, hi=None): # can't use a to specify default for hi
    hi = hi if hi is not None else len(a) # hi defaults to len(a)   
    pos = bisect_left(a,x,lo,hi)          # find insertion position
    return (pos if pos != hi and a[pos] == x else -1) # don't walk off the end    

  def add_trajectory_entry(self, state, action, args, timestamp):
    entry = TrajectoryEntry(state, action, args, timestamp)

    # Track action counts
    action = str(entry) 
    #if action not in self._action_counts:
      #self._action_counts[action] = 1
    #else:
      #self._action_counts[action] += 1

    self._call_trajectory.append(entry) # Log action for bookkeeping purposes
    self.add_state_observation(timestamp,state) # Record initial state (pre-function call)

    print "Added entry for action %s at time %g " % (action, timestamp)

    if len(self._call_trajectory) % self._checkpoint_interval == 0:
      print "Checkpointed at recorded action # %d" % len(self._call_trajectory)
      self.save_to_file('./checkpoint-%d' % len(self._call_trajectory))
      self.save_predicates()
      self.export_graph_json('./checkpoint-%d.js' % len(self._call_trajectory))
    return entry

  def populate_state_list(self):
    #  Adds states for each world state observed in state_history list
    new_states = []
    for state in self._state_history:
      if state not in self._states:
        self._states.append(state)
        new_states.append(state)
    return new_states

  def print_states(self):
    for idx,state in enumerate(self._states):
      print idx,':',str(state)
      
  def get_predicate_functions(self):
    funcs = []
    for predicate_tuple in self._planning_predicates:
      funcs.append(predicate_tuple[1])
    return funcs

  def compile_to_graph(self):
    '''
    Take all observed states and convert them into graph nodes, storing the result in self._graph
    '''        
    # Partial graph construction not 100% tested/functional yet - modifying edge transition probs not done.   
    self._last_timestamp_graphed = -1
    self._graph = nx.MultiDiGraph()
    self._states = []
    self._transition_counts = {} # [State][Action][State]
    #####################################  
  
    new_states = self.populate_state_list()
    for state in new_states:
      predicate_set = {}
      for predicate in Planning_Predicate.predicate_list:
        predicate_value = predicate[1](state.get_features())
        predicate_templates = predicate[2]
        if predicate_value is False: predicate_text = predicate_templates[1]
        else: predicate_text = predicate_templates[0]
        predicate_set[predicate[0]] = ( predicate_value, predicate_text )
      self._graph.add_node(str(state), reward=0., predicates=predicate_set)
      self._transition_counts[state] = {}
          
    action_attributed_intervals = [] # List of state indices that are covered by observed actions
    
    # Crawl each entry in the action history of the program's execution to set up the transition dynamics table
    for entry in self._call_trajectory:
      if entry._before_timestamp < self._last_timestamp_graphed and entry._after_timestamp < self._last_timestamp_graphed: continue

      # Connect all states between entry.before_state and entry.after_state
      prior_state_idx = self.get_nearest_state_idx(entry._before_timestamp)
      if entry._after_state == None:
        post_state_idx = prior_state_idx
        prior_state_idx = self._state_history.index(entry._before_state, prior_state_idx)
      else:
        post_state_idx = self.get_nearest_state_idx(entry._after_timestamp) # Repeated states exist- only search in relevant time interval
        prior_state_idx = self._state_history.index(entry._before_state, prior_state_idx, post_state_idx)
        
        next_timestamp_idx = post_state_idx
        while self._state_history_timestamps[next_timestamp_idx] == self._state_history_timestamps[post_state_idx] and next_timestamp_idx < len(self._state_history_timestamps): next_timestamp_idx += 1
          
        if entry._after_state not in self._state_history[post_state_idx:next_timestamp_idx]:
          continue
        
        
        post_state_idx = self._state_history.index(entry._after_state, post_state_idx, next_timestamp_idx)

      action_attributed_intervals.append((prior_state_idx, post_state_idx))

      # State history / timestamp based action links
      action = str(entry)      
      
      for i in range(prior_state_idx, post_state_idx):
        before_idx = i
        before_state = self._state_history[before_idx]
        after_idx = i+1
        after_state = self._state_history[after_idx]
      
        #if prior_state not in self._transition_counts:
          #self._transition_counts[prior_state] = {}
        if action not in self._transition_counts[before_state]:
          self._transition_counts[before_state][action] = {}
        if after_state not in self._transition_counts[before_state][action]:
          self._transition_counts[before_state][action][after_state] = 1
        else:
          self._transition_counts[before_state][action][after_state] += 1
        
    # Go through observed state history and connect temporally adjacent states that don't have an action edge
    start_state_idx = self.get_nearest_state_idx(self._last_timestamp_graphed + 1e-100)
    for i in xrange(start_state_idx,len(self._state_history)-1):
      before_state = self._state_history[i]
      after_state = self._state_history[i+1]

      # Don't need to include self-links
      if before_state == after_state: continue
        
      # Skip this transition if it's already explained by a recorded action
      transition_explained_by_action = False
      for interval in action_attributed_intervals:
        if i >= interval[0] and i+1 <= interval[1]:
          transition_explained_by_action = True
          break
      if transition_explained_by_action is True: continue
        
      # Don't add a non-action transition if an action-based transition exists
      # Possible issue -- if an action falsely attributes a state change, won't be able to detect it with this.
      # if self._graph.has_edge(str(before_state), str(after_state)): continue
        
      # Add a None transition between states
      if None not in self._transition_counts[before_state]: self._transition_counts[before_state][None] = {}
      if after_state not in self._transition_counts[before_state][None]: self._transition_counts[before_state][None][after_state] = 0
      self._transition_counts[before_state][None][after_state] += 1
    
    # Add edges
    for before_state in self._transition_counts:
      state_transitions = {} # [PostState][Action] = occurrences
      
      total_observed_transitions = 0
      for action in self._transition_counts[before_state]:
        for after_state in self._transition_counts[before_state][action]:
          if after_state not in state_transitions: state_transitions[after_state] = {}
          if action not in state_transitions[after_state]: state_transitions[after_state][action] = 0
          state_transitions[after_state][action] += 1
          total_observed_transitions += self._transition_counts[before_state][action][after_state]

      for after_state in state_transitions:
        total_transitions = 0
        for action in state_transitions[after_state]:
          total_transitions += state_transitions[after_state][action]
        for action in state_transitions[after_state]:
          state_transitions[after_state][action] /= float(total_transitions)
        edge = (str(before_state),str(after_state))
        if len(state_transitions[after_state]) > 1 or None not in state_transitions[after_state]:
          self._graph.add_edge(*edge, 
                               weight=state_transitions[after_state][max(state_transitions[after_state])],
                               action=state_transitions[after_state], reward=0.)
      '''
      for action in self._transition_counts[before_state]:
        for after_state in self._transition_counts[before_state][action]:
          weight = self._transition_counts[before_state][action][after_state] / total_observed_transitions
          edge = (str(before_state),str(after_state))
          self._graph.add_edge(*edge, weight=weight, reward=0., action=action)
      '''
    # Apply reward signals
    for reward in self._rewards_applied:
      if reward._timestamp > self._last_timestamp_graphed:
        self.apply_reward_to_transitions(reward._reward, reward._timestamp)
    
    self._last_timestamp_graphed = time.time()
    
    self._graph.predicates = [x[0] for x in Planning_Predicate.predicate_list]

  def get_graph_node(self, state_vector):
    if state_vector.__class__ == list:
      target_state_str = json.dumps(state_vector)
    elif state_vector.__class__ == State:
      target_state_str = str(state_vector)
    for node in self._graph.nodes(data=True):
      if node[0] == target_state_str:
        return node
    return None
    
  def store_reward(self, reward, timestamp):
    r = RewardEntry(reward, timestamp)
    print "Stored reward of %g at time %g" % (reward, timestamp)
    self._rewards_applied.append(r)
    return r
    
  def apply_reward_to_transitions(self, reward, apply_at_timestamp, decay_rate=0.9, learning_rate=0.05, cutoff_threshold=0.1):
    #self.compile_to_graph()
    
    state_idx_at_event = self.get_nearest_state_idx(apply_at_timestamp + 1e-5)
    if state_idx_at_event is None: 
      return False

    cur_action_idx = len(self._call_trajectory)-1 # Find preceeding action
    while self._call_trajectory[cur_action_idx]._after_timestamp > apply_at_timestamp and cur_action_idx > 0:
      cur_action_idx -= 1

    if len(self._discontinuities) == 0:
      discontinuity_idx = None
    else:
      discontinuity_idx = max(0, bisect_left(self._discontinuities, apply_at_timestamp) - 1)
      
    cur_state_idx = state_idx_at_event
    cur_reward = reward
    last_rewarded_state = None
    # Backtrack through state history, applying negative reward.
    while cur_state_idx > 0:
      
      if discontinuity_idx is not None and self._state_history_timestamps[cur_state_idx] < self._discontinuities[discontinuity_idx]:
        print "Hit discontinuity at %g at state %s! Started applying at state entry %d, stopped at state entry %d." % (self._discontinuities[discontinuity_idx], str(self._state_history[cur_state_idx]), state_idx_at_event, cur_state_idx)
        return True
      
      if abs(cur_reward) < cutoff_threshold:
        return True     
      
      graph_node = self.get_graph_node(self._state_history[cur_state_idx])
      if graph_node is None:
        raise Exception('Non-existent graph node pulled from state history')

      if graph_node != last_rewarded_state:
        # Perform value update
        state_reward = graph_node[1]['reward']
        new_reward = (1. - learning_rate) * state_reward + learning_rate * cur_reward
        print "Set new reward of %g to state %s." % (new_reward, str(graph_node[0]))
        graph_node[1]['reward'] = new_reward
        last_rewarded_state = graph_node
      # Apply decay at each 'action' timestamp
      while cur_action_idx >= 0 and self._call_trajectory[cur_action_idx]._after_timestamp >= self._state_history_timestamps[cur_state_idx]:
        cur_reward *= decay_rate
        cur_action_idx -= 1

      # If current reward/penalty magnitude < cutoff_threshold, then return
      cur_state_idx -= 1

    return True
  
  def add_discontinuity(self, timestamp):
    if timestamp not in self._discontinuities:
      self._discontinuities.append(timestamp)
    return



  #################################    
  '''
  Question Resolution and Response Functions
  '''
  #################################    
  def determine_question_type(self, question_text):
    action_list_template = ["What are your actions", "What do you do"]
    current_state_template = ["Where are you"]
    action_description_templates = ["When will you ", "When do you "]
    state_description_templates = ["What will you do when ", "What do you do when "]
    difference_description_templates = ["Why didn't you ", "Why didn't you do ", "Why aren't you doing "]
      

    # Array of (Type, Template) pairs
    question_templates = [ ('action_summary', action_description_templates),
                           ('current_status', current_state_template),
                           ('state_summary', state_description_templates),
                           ('difference_summary', difference_description_templates),
                           ('list_actions', action_list_template) ]
    question_type = None    
    argument_text = '';
    
    for q_type, q_templates in question_templates:
      if question_type is not None: break
      for template in q_templates:
        if question_text[:len(template)] == template:
          question_type = q_type
          argument_text = question_text[len(template):]
          argument_text = argument_text.replace(" your ", " my ")
          break
        
        
    # List question types
    if question_type is None:
      question_template_str = [entry[1][0] for entry in question_templates]
      for idx,entry in enumerate(question_template_str):
        if entry[-1] != ' ': question_template_str[idx] = entry + "?"
        else:  question_template_str[idx] = entry + "____?"
      
      return 'list_questions', "The question types I can answer are: " + "\n".join(question_template_str)


    return question_type, argument_text
    
    
  def find_mentioned_actions(self, text):
    '''
    @param text Block of text possibly containing comma-separated action descriptions
    '''
    actions = []    
    unresolved_args = []
    possible_args = text.split(',')
    
    for arg in possible_args:
      resolved = False
      for action in self._actions:
        action_obj = self._actions[action]
        if action_obj._name in arg or action_obj._nlp_name in arg: 
          resolved = True 
          actions.append(action_obj)
      if resolved is False:
        unresolved_args.append(arg)
      
    return actions, unresolved_args
    
       
  def answer_question(self, question_text):
    # get start time
    time_a = datetime.datetime.now()

    # Map question_text to a question type
    question_text = re.sub('[!@#$?.]','',question_text)
    question_text = question_text.strip()
    question_type, argument_text = self.determine_question_type(question_text)
  
    # Extract argument text for each particular type
    if question_type == 'action_summary': # When do you X?
      descriptions = self.describe_action_clusters(self._graph, argument_text)
    elif question_type == 'state_summary': # What do you do if Y?
      descriptions = self.describe_state_behaviors(self._graph, self._planning_predicates, argument_text)
    elif question_type == 'difference_summary': # Why didn't you X?
      print "graph"
      print self._graph
      print "planning predicates"
      print self._planning_predicates
      print "planning argument text"
      print argument_text
      descriptions = self.explain_unexpected_behavior(self._graph, self._state_history, self._planning_predicates, argument_text)
    elif question_type == 'current_status': # Where are you?
      descriptions = "I am currently in state %s" % str(self._state_history[-1])
    elif question_type == 'list_actions': # Give back list of action descriptors
      descriptions = self.list_actions(self._actions)
    elif question_type == 'list_questions': # Give back list of question templates
      descriptions = argument_text
    elif question_type is None:
      descriptions = "I didn't understand the question."
    else:
      raise Error('Unhandled question type found')
    # Get DNF-equivalent description from question-specific functions
    
    time_b = datetime.datetime.now()
    time_delta = time_b - time_a
    # Return answer
    return descriptions + "\n Time to answer (s): " + str(time_delta.total_seconds())

  #################################
  '''
    Question: What do you do?
    Type: 'list_actions'
    List NLP descriptions for each action
  '''
  ################################
  def list_actions(self,action_list):
    '''
    @param action_list A dictionary of PCCA_Function objects with function_name keys
    '''
    descriptions = []
    for action in action_list:
      if hasattr(action_list[action], '_nlp_name') and len(action_list[action]._nlp_name) > 0:
        descriptions.append(action_list[action]._nlp_name)
      else:
        descriptions.append(action_list[action]._name)
        
    return 'I can ' + ', '.join(descriptions)







  #################################    
  '''
    Question: When will you {action_list}?
    Type: 'action_summary'
    Find state clusters where action_list elements are the most probable action
  '''
  #################################   

  # Helper function for identify_action_clusters
  def get_most_probable_action(self,state):
    '''
      state - State object
      return - Tuple of max action (#, Action), action_counts dict
    '''
    counts = {}
    max_count = (0, None)

    if len(self._transition_counts) == 0:
      raise UserWarning('Transition counts dictionary not initialized when get_most_probable_action called.')
    
    for action in self._transition_counts[state]:
      count = 0
      for next_state in self._transition_counts[state][action]:
        count += self._transition_counts[state][action][next_state]
      counts[action] = count
      if count > max_count[0]:
        max_count = (count, action)
    # print "State " + str(state)
    # print "Most probable action computation"
    # print str(max_count)
    # print str(counts)    
    return max_count, counts
  
  def identify_action_clusters(self, state_list=None):
    # Given graph, return a list of state clusters where particular actions are executed
    # USE CASE: when will you do X?
  
    # Find top-1 action for each state, remove all other edges from graph
    # For each known action... make a state-cluster
    #   * Iterate through all graph edges, looking for action label matching target action 
    #     * Remove matching edges from edge list and add origin state to current_action_cluster
    #
  
    if state_list is None: state_list = self._states  
  
    action_clusters = {} # Key=Action_name, Value=list of states
    
    for state in state_list:
      action, _ = self.get_most_probable_action(state)
      if action[1] is None: continue
      # TODO: Can set tiers of answers: "Always / Mostly / Sometimes / Rarely" corresponding to probability thresholds
      action_str = action[1]
      if action_str not in action_clusters: action_clusters[action_str] = []
      action_clusters[action_str].append(state)                 

    #self._action_clusters = action_clusters  # Cache result in case it's needed later
    return action_clusters

  def describe_action_clusters(self, graph, argument_text):
    '''
      @param graph Behavioral Graph
      @param argument_text Comma-separated list of action names
    '''
    descriptions = {}
    
    actions, unresolved_args = self.find_mentioned_actions(argument_text)      

    for arg in unresolved_args:
      print "WARNING: Unresolved argument '%s'" % arg
    
    if len(actions) == 0:
      return "I didn't recognize any actions that you asked about"
    
    descriptions = self.generate_action_cluster_descriptions(graph=graph, state_list=self._states, action_list=actions, threshold=len(actions))

    individual_descriptions = []
    for action_name in descriptions:
      individual_descriptions.append('I do %s when %s.' % (action_name, descriptions[action_name]))
      
    return ' '.join(individual_descriptions)

  #################################    
  '''
    Question: What will you do when {state_description}?
    Type: 'state_summary'
    Find action clusters within all states covered by state_description
  '''
  #################################   
  def generate_action_cluster_descriptions(self, graph, state_list, action_list=None, threshold=5):
    '''
      @param graph Behavioral graph
      @param state_list list of states to include when summarizing action policy    
      @param threshold Maximum number of action clusters to include in summary
    '''
    action_clusters = self.identify_action_clusters(state_list)    
    '''
    Build explanations for each state list
    '''
    descriptions = {}
    for action_type in action_clusters:
      if type(action_list) is list and PCCA_Function.from_string(self._actions.values(),action_type) not in action_list: continue
      descriptions[action_type] = self.solve_for_state_description(state_list=action_clusters[action_type], recompute_predicates=True, predicates_list=self.get_predicate_functions())

    return descriptions

  def resolve_predicate_list_to_state_list(self, predicate_tuple_list):
    '''
      @param predicate_tuple A tuple containing two lists of boolean state classifiers: (true_concepts, false_concepts)
      @param description_text A text blob describing the target set of states
    '''
    state_list = []

    for predicate_tuple_entry in predicate_tuple_list:
      true_concepts, false_concepts = predicate_tuple_entry
        
      # If no known predicates were described, return empty.
      if len(true_concepts) == 0 and len(false_concepts) == 0: return state_list
  
      for state in self._states:
        # Verify that state matches region of interest
        valid = True
        #print "State: " + str(state)
        target_values = [True, False]
        lists = [true_concepts, false_concepts]        

        for concept_list, target_value in zip(lists, target_values):
          for concept in concept_list:
            concept_func = concept[1]
            if concept_func(state.get_features()) is not target_value: 
              valid = False          
              break
            if valid is False: break              
          if valid is False: break

        if valid is True:
          state_list.append(state)

    return state_list


  def resolve_concept_description_to_concepts(self, predicates, description_text):
    true_concepts = []
    false_concepts = []
    original_description_text = copy.copy(description_text)

    # Populate true/false/indifferent predicate lists
    for predicate in predicates:
      positive_location = description_text.find(predicate[2][0])
      negative_location = description_text.find(predicate[2][1])
      use_positive = positive_location != -1
      use_negative = negative_location != -1
      if use_positive is True and use_negative is True:
        use_positive = len(predicate[2][0]) >= len(predicate[2][1])
        use_negative = not use_positive

      # Remove used predicate description text from input string
      if use_positive is True: 
        true_concepts.append(predicate)
        description_text = description_text[:positive_location] + description_text[positive_location+len(predicate[2][0]):]
      elif use_negative is True: 
        false_concepts.append(predicate)
        description_text = description_text[:negative_location] + description_text[negative_location+len(predicate[2][1]):]
    
    print "Resolved %s to %s, %s" % (original_description_text, str(true_concepts), str(false_concepts))
    return true_concepts, false_concepts  
    
  def find_mentioned_concepts(self, predicates, text):
    '''
    @param predicates - List of tuples of (func_name, func, (positive_string, negative_string)) as in self._planning_predicates
    @param text Block of text possibly containing comma-separated, 'and'-separated, and 'or'-separated clauses of concepts
    '''
    term_delimiters = [',', ' and ']
    term_split_regex = '|'.join(map(re.escape, term_delimiters))

    concepts = [] # DNF list of concepts. Elements are (true concept list, false concept list) for each clause
    
    clauses = text.split(' or ')
    for clause in clauses:      
      print "--Clause: " + str(clause)
      terms = re.split(term_split_regex, clause)
      print "-- TERMS: " + str(terms)
      true_concepts = []
      false_concepts = []
      for term in terms:
        # Match against list of concepts' true and false values
        tc, fc = self.resolve_concept_description_to_concepts(predicates, term)
        true_concepts.extend(tc)
        false_concepts.extend(fc)
      concepts.append( (true_concepts, false_concepts) )
    return concepts

  def describe_state_behaviors(self, graph, predicates, argument_text):
    dnf_clauses = self.find_mentioned_concepts(predicates, argument_text)
    state_list = self.resolve_predicate_list_to_state_list(dnf_clauses)

    #print "State list: " + ', '.join([str(i.get_features()) for i in state_list])

    if len(state_list) == 0:
      print "Warning: argument_text matched no states."
      return "No states that I've seen match that description."
      
    descriptions = self.generate_action_cluster_descriptions(graph=graph, state_list=state_list, action_list=None, threshold=5)

    individual_descriptions = []    
    for action_name in descriptions:
      individual_descriptions.append('I do %s when %s.' % (action_name, descriptions[action_name]))
    
    description = ' '.join(individual_descriptions)
    return description

  #################################    
  '''
    Question: Why aren't you doing {action}?
    Type: 'difference_summary'
    Find and describe states nearby the current state where {action} is the most probable choice.
  '''
  #################################   
  def explain_unexpected_behavior(self, graph, state_history, predicates_list, argument_text):
    '''
      @param graph Behavioral graph
      @param state_history Agent state history
      @param predicates_list List of boolean state classifiers
      @param argument_text String containing the action that was expected to occur
    '''                             
    if len(state_history) <= 2:
      raise Exception('No state history given... need at least visit two states!')

    expected_actions, unresolved_args = self.find_mentioned_actions(argument_text)
    if len(expected_actions) == 0:
        raise Exception('No actions mentioned in question argument')
      
    expected_action = expected_actions[0] # Only take the first action found ... question doesn't support multiple arguments.

    # Convert PCCA_Function to function description.
    # e.g. move_north()=None
    expected_action = expected_action._name + "()=None"

    # current_state = state_history[-1] #StateVarRef.get_state_snapshot()
    distance_threshold = 3 # Maximum distance away to search for states with policy(state)==expected_action. 
                           # Distance is measured in terms of graph distance here (#vertices away), but in the future could be sampled using a distance vector with max-radius for each feature

    last_state = state_history[-2]
    action, _ = self.get_most_probable_action(last_state)
    action_name = action[1]
    print "Most probable action: " + str(action_name)
    if action_name == expected_action: 
      return "I did %s" % action_name
    


    # Get list of all states within distance_threshold of last state
    target_list = []
    open_list = [(last_state,0)]
    
    print "Open List"
    print open_list

    while len(open_list) > 0:
      state, depth = open_list.pop()
      print state
      if depth >= distance_threshold: continue
      target_list.append(state)        

      for edge in graph.in_edges(state):
        print "Edge: " + str(edge[0])

      neighbors = [edge[0] for edge in graph.in_edges(state)] + [edge[1] for edge in graph.edges(state)]

      for neighbor_state in neighbors:
        if neighbor_state not in target_list:
            open_list.append((neighbor_state,depth+1))

    print target_list
    del target_list[0] # Remove last_state from target list

    # Get action clusters within nearby states
    action_region = []
    non_action_region = []

    for state in target_list:
      action, _ = self.get_most_probable_action(state)
      print "Most probable action for state " + str(state)
      print action
      action_name = action[1]
      if action_name == expected_action:
        # TODO - debug
        action_region.append(state)
      else:
        non_action_region.append(state)
    
    # Get predicate explanations for action region
    total_state_list = action_region + non_action_region + [last_state]
    print "Action Region"
    print action_region
    print "Non-action region"
    print non_action_region

    action_region_description, time_in_min_a = self.solve_for_state_description(action_region)
    current_region_description, time_in_min_b = self.solve_for_state_description([last_state], action_region + non_action_region)
    descriptions = { 'action_region': action_region_description, 'current_region': current_region_description }    
    
    print "ACTION REGION"
    print action_region_description

    description = "I perform %s when %s, but in the last state I was %s. Time in minimzation step: %s" % (expected_action, descriptions['action_region'], descriptions['current_region'], str((time_in_min_b + time_in_min_a))) # TODO: Improve return of this question by providing proper diff / removing predicates common to both action/current regions
    
    return description

  ####################################




  def get_state_predicate_value(self,state,predicates_list=None):
    '''
      state - JSON string of features or a State object
      predicates_list - list of predicate functions
      
      returns - integer encoding predicate truth values, least-significant-bit first
    '''
    if predicates_list is None:
      predicates_list = Planning_Predicate.get_predicate_functions()
    if state.__class__ == str:
      state = json.loads(state) # JSON Feature Vector String passed in as state
    elif state.__class__ == State:
      state = state.get_features() # Get feature vector from state object
      
    val = 0
    for idx in range(len(predicates_list)):
      predicate_val = predicates_list[idx](state)
      val |= predicate_val << idx

    return val
  
  def state_predicate_value_matches_qm_string(self, state_val, qm_string):
    for idx in range(len(qm_string)):
      if state_val & (1 << idx) != 0: # Predicate idx is True
        if qm_string[idx] == '0':  # Any value but False will match (E.g., True or X)
          return False
      else: # Predicate idx is False
        if qm_string[idx] == '1':  # Any value but True will match (E.g., True or X)
          return False
    return True
  
  def state_matches_minterm_list(self,state, minterm_list, predicates_list=None):
    '''
    state - State object
    minterm_list - list of strings in the language [0,1,X]* of length log2(len(predicates_list))
    predicates_list - list of boolean functions
    '''
    for minterm in minterm_list:
      if self.state_predicate_value_matches_qm_string(self.get_state_predicate_value(state,predicates_list), minterm): return True
    return False


  def solve_for_state_description(self, state_list, total_state_list=None, recompute_predicates=False, predicates_list=None):
    '''
      state_list - List of states to describe
      total_state_list - List of states to consider when forming cover
    '''

    # Get predicate explanations for action region
    cover, overstatement, understatement, time_tmp = self.solve_for_state_description_cover(state_list, total_state_list, recompute_predicates, predicates_list)
    
    explanations = []
    
    values, predicate_funcs = cover
    for clause in values:
      clause_explanation = []
      for idx, predicate_value in enumerate(clause):
        if predicate_value == '1':   clause_explanation.append(self._planning_predicates[idx][2][0])
        elif predicate_value == '0':  clause_explanation.append(self._planning_predicates[idx][2][1])
      
      clause_summary = ' and '.join(clause_explanation)
      if clause_summary not in explanations:
        explanations.append(clause_summary)
      
    return ' ---or--- '.join(explanations), time_tmp
   
    
  def solve_for_state_description_cover(self, state_list, total_state_list=None, recompute_predicates=False, predicates_list=None):
    '''
    # Solves for the best covering set of predicates that describes a list of states

    ##### ARGS #####
    state_list : List of states to include in description
    total_state_list : List of states to consider in cover solution
    recompute_predicates : Default False - uses predicate values from graph construction time. True to re-evaluate.
    predicates_list : Default None - If not specified and recompute_predicates is True, use Planning_Predicate list. Otherwise recompute predicates for node_list using the functions in predicates_list

    ##### RETURNS #####
    Returns predicate cover set, precision measures -- overstatement (#fp / #tp), understatement (#fn / #tp)
    '''
    overstatement = 0.
    understatement = 0.

    include_table = {}
    exclude_table = {}   
    
    positive_minterms = []
    negative_minterms = []
    dc_minterms = []

    if predicates_list is None:
      predicates_list = self.get_predicate_functions()
    if total_state_list is None:
      total_state_list = self._states 
    '''
    for state in state_list:
      if recompute_predicates is True:
        state.compute_predicates(predicates_list)
      predicates, predicates_list = state.get_predicates()  # List of boolean values
      val = 0
      for idx in range(len(predicates)):
        val |= predicates[idx] << idx      
        
      if state in state_list:
        if val not in include_table: 
          include_table[val] = 1
        else: 
          include_table[val] += 1

    nonspec_is_negative = True
    '''
    nonspec_is_negative = False   

    for state in total_state_list:
      print type(state)
      if type(state) == str:
        state = ast.literal_eval(state) # in writing to graph, states are recorded as strings, e.g. "[[0,0]]"
        print state
        state = State(state, self.get_predicate_functions())
      if recompute_predicates is True:
        state.compute_predicates(self.get_predicate_functions())
      predicates, predicates_list = state.get_predicates()  # List of boolean values
      val = 0
      for idx in range(len(predicates)):
        val |= predicates[idx] << idx      
        
      if state in state_list:
        if val not in include_table: 
          include_table[val] = 1
        else: 
          include_table[val] += 1
      else:        
        if val not in exclude_table: 
          exclude_table[val] = 1
        else: 
          exclude_table[val] += 1

    print "Include: ", include_table
    print "Exclude: ", exclude_table

    # IDEA: Can use frequency of minterms in truth_table as measure for most important minterms
        
    # Collect all minterms
    positive_minterms = include_table.keys()
    for minterm in include_table:
      print "Added positive minterm: %s (%s)" % (str(minterm), b2s(minterm,len(predicates_list)))
      if minterm in exclude_table:
        print "WARNING: positive minterm found in negative minterm table. Removing from negative minterm table."  # Minterms can't be positive and negative.
        del exclude_table[minterm]


    if nonspec_is_negative is True:
      negative_minterms = [i for i in xrange(2**len(predicates))]
      for minterm in positive_minterms:
        negative_minterms.remove(minterm)
    else:            
      negative_minterms = exclude_table.keys()
      '''
      for minterm in exclude_table:
        inverse = ~minterm & 2**len(predicates_list) - 1
        if inverse not in positive_minterms: positive_minterms.append(inverse)
        print "Added negative minterm: %s -> %s (%s)" % (str(minterm), str(inverse), b2s(inverse,len(predicates_list))) 
      '''
    time1 = datetime.datetime.now()
    # Retrieve minimized formula describing state region:
    # qm_minimization is string in [0,1,X]* that indexes into final_predicate_list
    #   - first element in qm_minimization corresponds to last element of final_predicate_list
    qm_minimization, qm_predicate_list = self.perform_boolean_minimization(predicates_list, positive_minterms, negative_minterms, dc_minterms)    
    final_predicate_minimization = []
    for minterm in qm_minimization:
      final_predicate_minimization.append(minterm[::-1])
    print "Initial QM Minimization: %s" % qm_minimization[::-1]
    
    final_predicate_list = qm_predicate_list
    time2 = datetime.datetime.now()
    time_diff = time2-time1
    '''
    # TODO: If predicate is DC in -ALL- clauses, remove it from the description
    
    for qm_minimization_str in qm_minimization:
      for char,predicate in zip(qm_minimization_str, qm_predicate_list):
        if char == 'X': continue  # Skip "Don't Care" entries.
        final_predicate_minimization = final_predicate_minimization + char
        final_predicate_list.append(predicate)
    '''
    
    print "Final QM Minimization: %s" % final_predicate_minimization
    print "Predicates: %s" % str(predicates_list)

    state_description = (final_predicate_minimization, final_predicate_list)
    
    overstatement, understatement = self.evaluate_state_description(state_list, final_predicate_minimization, final_predicate_list)
    
    print "Overstatement %g, Understatement %g." % (overstatement, understatement)
    return state_description, overstatement, understatement, time_diff.total_seconds()
  
  def evaluate_state_description(self, state_list, minterm_list, predicate_list):
    '''
    Given a state_list, encoded DNF formula, and predicate list, return overstatement and understatement measurement within state space
    # At this point, binary_description_list should be an integer with len(predicate_list) bits.
    '''    

    if len(minterm_list) > 0 and len(minterm_list[0]) != len(predicate_list):
      raise Exception('%d minterms for %d predicates provided!' % (len(minterm_list[0]), len(predicate_list)))
    
    
    #print "State_list: "
    #for state in state_list: print str(state)    
    
    # Count states matching qm_minimization in self._graph node list
    matching_states_list = []
    for state in self._states:
      if self.state_matches_minterm_list(state, minterm_list, predicate_list) is True:
        #print "Matched state " + str(state)
        matching_states_list.append(state)
        
    # False Positives: states in matching_states_list not found in state_list
    false_positives = 0
    for state in matching_states_list:
      if state not in state_list:
        #print "FP: State %s not in state_list" % str(state)
        false_positives += 1
    
    # False Negatives: state_list states not found in matching_states_list
    false_negatives = 0
    for state in state_list:
      if state not in matching_states_list:
        #print "FN: State %s not in state_list" % str(state)
        false_negatives += 1

    overstatement = float(false_positives) / float(len(self._states))
    understatement = float(false_negatives) / float(len(self._states))
    return overstatement, understatement
    
  def perform_boolean_minimization(self, master_predicate_list, include_minterms, exclude_minterms, dc_minterms=[]):
    '''
    # Runs Quine-McCluskey algorithm on set of minterms
    # Args:
    #   master_predicate_list - list of possible predicates: e.g., [p1, p2, p3, ..., pn]    
    #   minterms - list of predicate truth values -- e.g., [ 10010, 01010, 01111 ] -- for each included state. 
    #     - len(minterms[0]) == len(master_predicate_list)
    # Returns:
    #   Minimized boolean expression: e.g., [ [True, None, None, False, None], [False, False, None, True, None] ]
    
    # TODO: Possible optimization by minimizing the master_predicate_list size before running Quine-McCluskey
    '''
    
    relaxation = dc_minterms
    
    for minterm in relaxation:
      exclude_minterms.remove(minterm)

    print 'Include: %d, Exclude: %d' % (len(include_minterms), len(exclude_minterms))
    qm_minimization = qm(ones=include_minterms, zeros=exclude_minterms, dc=relaxation)
         
    return qm_minimization, master_predicate_list
   
  def convert_predicate_list_into_text(self, minterm_list, predicate_list):
    '''
       minterm_list: list of strings encoding of predicate truth values [0,1,X]
       predicate_list: list of predicate functions used to construct predicate_values (bits in values index into list)
       ---
       returns explanation: text string combining the predicate value descriptions
    '''
    explanation = ''  # Final text output from the predicate list
    predicate_tuples = []  # List of the (func_name, func, text) tuples corresponding to predicate_list
    
    # Lookup full predicate tuple (name, function, positive-text, negative-text)
    for predicate_func in predicate_list:
      for predicate_tuple in self._planning_predicates:
        if predicate_tuple[1] == predicate_func:
          predicate_tuples.append(predicate_tuple)
          break
    
    for minterm in minterm_list:
      predicate_texts = []
      for idx in range(len(predicate_tuples)):
        if minterm[idx] == 'X': continue  # Skip Don't Care values
        predicate_value = minterm[idx] == '1'
        if predicate_value is True:
          predicate_texts.append(predicate_tuples[idx][2][0])  # Add positive text
        else:
          predicate_texts.append(predicate_tuples[idx][2][1])  # Add negative text
    
      explanation += '], OR [' + ' and '.join(predicate_texts)
    
    return explanation[7:]  # Contains the 'when' portion of the policy explanation
     

  ##################################################################################
  '''
    PCCA Mode Identification
  '''
  ##################################################################################
  # TODO
  def identify_clusters(self, graph):
    # Given graph, return a list of nominal and failure clusters with node memberships
    # USE CASE: Build PCCA to use for identifying mode frontiers
    
    
    ##################################################################################
    ################# CLASSIFY NODES AS EITHER NOMINAL OR FAILURE ####################    
    ##################################################################################
    nominal_nodes = []
    failure_nodes = []
    FAILURE_THRESHOLD = -1.
  
    # TODO: Look into transitioning to a two-threshold model with 3 classes: Definite FAILURE, Uncertain, Definite NOMINAL (Classification with rejection option)
    for node in graph.nodes(data=True):
      if node[1]['reward'] < FAILURE_THRESHOLD:
        failure_nodes.append(node[0]) # Only append state identifier to list
      else:
        nominal_nodes.append(node[0]) # Only append state identifier to list

    ##################################################################################
    ########################### SUBCLASSIFY FAILURE MODES ############################    
    ##################################################################################
    
        
    nominal_clusters = [] # List of lists
    failure_clusters = [] # List of lists
    # Run optimization on cluster memberships, with succinctness-based objective function        
    
    return nominal_clusters, failure_clusters


  # TODO
  def solve_for_cluster_transition_description(self, graph, from_cluster_nodes, to_cluster_nodes):
    # Return the best covering set describing the transition between two clusters --
    # * Describe the delta between linked states
    # USE CASE: When does the robot enter gripper failure? -- Solve_for_state_description for all nodes entering to_cluster that aren't in to_cluster
    # USE CASE: What is gripper failure? -- Maybe use solve_for_state_description to compare difference between TO and FROM cluster?   

    predicate_list = [] # List of lists for DNF-form description. Outer lists are OR'd together, inner lists are AND'd together.
    return predicate_list


  
  # TODO
  def get_cluster_transition_edges(self, graph, from_cluster_nodes, to_cluster_nodes):
    # Returns the set of edges that traverse any subset of from_cluster_nodes to to_cluster_nodes
    # Also returns sets of nodes not included in the from_set and to_set

    # Returned variables
    transition_edges = []
    from_inactive_nodes = {}
    from_frontier_nodes = {}
    to_inactive_nodes = {}
    to_frontier_nodes = {}
    
    # Isolate all edges that involve from or to cluster:
    for edge in self._graph.edges(data=True):
      edge_from, edge_to = edge[0], edge[1]
      if edge_from in from_cluster_nodes and edge_to in to_cluster_nodes:
        if edge_from not in from_frontier_nodes: from_frontier_nodes[edge_from] = 0
        from_frontier_nodes[edge_from] += 1
        # TODO!
    # Identify all 'frontier' nodes between the clusters:
    
    return transition_edges, from_inactive_nodes, from_frontier_nodes, to_inactive_nodes, to_frontier_nodes


def test():
  p = PCCA.load_from_file('../Simulation/checkpoint-500')
  goal = p._states[9]
  descr = p.solve_for_state_description([goal], None, True, p.get_predicate_functions())
  print descr


main_pcca = PCCA()
PCCA_Function.main_pcca = main_pcca
