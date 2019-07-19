# -*- coding: utf-8 -*-
"""
Simulator for Chopsticks game example
"""

import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Code/'))
sys.path.insert(0,lib_path)

import random
import tornado.web
import tornado.websocket
import tornado.ioloop
import argparse
from pcca import *
from pcca_decorator import *

global global_pcca

class World:
  
  def __init__(self): 
    self.piles = [[1,1],[1,1]]
    self.create_predicate_functions()

  def create_predicate_functions(self):
    @Planning_Predicate("my total is even")
    def even_total(state):
      if (state[0][0][0] + state[0][0][1]) % 2 == 0:
        return True
      return False

    # @Planning_Predicate("my hands have the same number")
    # def same_hands(state):
    #   if (state[0][0][0] == state[0][0][1]):
    #     return True
    #   return False

    @Planning_Predicate("my left hand is out")
    def left_out(state):
      if (state[0][0][0] == 0):
        return True
      return False

    @Planning_Predicate("my right hand is out")
    def right_out(state):
      if (state[0][0][1] == 0):
        return True
      return False

    @Planning_Predicate("my left hand outs my opponent's left hand")
    def can_empty_opponent_left_lhs(state):
      if (state[0][0][0] + state[0][1][0]) % 5 == 0 and \
          state[0][0][0] != 0:
        return True
      return False

    @Planning_Predicate("my right hand outs my opponent's left hand")
    def can_empty_opponent_right_lhs(state):
      if (state[0][0][1] + state[0][1][0]) % 5 == 0 and \
          state[0][0][1] != 0:
        return True
      return False
    
    @Planning_Predicate("my left hand outs my opponent's right hand")
    def can_empty_opponent_left_rhs(state):
      if (state[0][0][0] + state[0][1][1]) % 5 == 0 and \
          state[0][0][0] != 0:
        return True
      return False

    @Planning_Predicate("my right hand outs my opponent's right hand")
    def can_empty_opponent_right_rhs(state):
      if (state[0][0][1] + state[0][1][1]) % 5 == 0 and \
          state[0][0][1] != 0:
        return True
      return False

    # @Planning_Predicate("my left hand is out")
    # def left_hand_out(state):
    #   if state[0][0][0] == 0:
    #     return True
    #   return False

    # @Planning_Predicate("my right hand is out")
    # def right_hand_out(state):
    #   if state[0][0][1] == 0:
    #     return True
    #   return False

    # @Planning_Predicate("I can out one of my opponent's hands")
    # def cannot_empty_opponent_piles(state):
    #   if (((state[0][0][0] + state[0][1][0]) % 5 == 0 and \
    #      (state[0][0][0] + state[0][1][1]) % 5 == 0) and (state[0][0][0] != 0)) and \
    #      (((state[0][0][1] + state[0][1][0]) % 5 == 0 and \
    #      (state[0][0][1] + state[0][1][1]) % 5 == 0) and (state[0][0][1] != 0)):
    #     return True
    #   return False

  def get_agent_piles(self):
    return [self.piles[0][0], self.piles[0][1]]

  def get_agent_left_pile(self):
    return self.piles[0][0]

  def get_agent_right_pile(self):
    return self.piles[0][1]

  def get_opponent_piles(self):
    return [self.piles[1][0], self.piles[1][1]]

  def get_opponent_left_pile(self):
    return self.piles[1][0]

  def get_opponent_right_pile(self):
    return self.piles[1][1]

  def get_all_piles(self):
    #[[piles[0][0], piles[0][1]], [piles[1][0], piles[1][1]]]
    return self.piles
    
  def achieve_goal(self):
    if self.check_goal():
      return True
    return False

  def check_goal(self):
    """
    Check if both opponent piles are empty 
    Parameters
    ----------
    self : World 
    Returns
    -------
    bool : True iff both opponent piles are empty 
    """
    if self.piles[1][0] == 0 and self.piles[1][1] == 0:
      return True
    return False

  def take_down_one(self):
    if self.piles[1][0] == 0 or self.piles[1][1] == 0:
      return True
    return False 

  def check_losing(self):
    """
    Check if both opponent piles are empty 
    Parameters
    ----------
    self : World 
    Returns
    -------
    bool : True iff both opponent piles are empty 
    """
    if self.piles[0][0] == 0 and self.piles[0][1] == 0:
      return True
    return False

  def check_valid_moves(self, current_config):
    valid_configurations = []

    valid_configurations.append([self.current_config[0], [self.current_config[1][0] + self.current_config[0][0], self.current_config[1][1]]])
    valid_configurations.append([self.current_config[0], [self.current_config[1][0] + self.current_config[0][1], self.current_config[1][1]]])
    valid_configurations.append([self.current_config[0], [self.current_config[1][0], self.current_config[1][1] + self.current_config[0][0]]])
    valid_configurations.append([self.current_config[0], [self.current_config[1][0], self.current_config[1][1] + self.current_config[0][1]]])

    if (self.current_config[0][0] + self.current_config[0][1]) % 2 == 0:
      split_piles = (self.current_config[0][0] + self.current_config[0][1]) / 2
      valid_configurations.append([[split_piles, split_piles], self.current_config[1]])

    return valid_configurations

  def update_opponent_piles(self, agent, update_instructions):    
    agent_pile = update_instructions[0]
    opponent_pile = update_instructions[1]

    print "update opponent starting " + str(self.piles)
    if agent_pile == 0 and opponent_pile == 0:
      proposed_piles = [[self.piles[0][0],self.piles[0][1]], [(self.piles[1][0] + self.piles[0][0]) % 5, self.piles[1][1]]]
    elif agent_pile == 1 and opponent_pile == 0:
      proposed_piles = [[self.piles[0][0],self.piles[0][1]], [(self.piles[1][0] + self.piles[0][1]) % 5, self.piles[1][1]]]
    elif agent_pile == 0 and opponent_pile == 1:
      proposed_piles = [[self.piles[0][0],self.piles[0][1]], [self.piles[1][0], (self.piles[1][1] + self.piles[0][0]) % 5]]
    elif agent_pile == 1 and opponent_pile == 1: 
      proposed_piles = [[self.piles[0][0],self.piles[0][1]], [self.piles[1][0], (self.piles[1][1] + self.piles[0][1]) % 5]]
    print "update opponent ending " + str(proposed_piles)

    self.piles = proposed_piles
    return proposed_piles

  def split_piles(self, agent):
    total_to_split = self.piles[0][0] + self.piles[0][1]
    if total_to_split % 2 != 0:
      print "Cannot split uneven piles" 
      return self.get_all_piles()
    else: 
      proposed_piles = [[total_to_split / 2, total_to_split / 2], [self.piles[1][0],self.piles[1][1]]]
      self.piles = proposed_piles
      return proposed_piles

  def update_agent_piles(self, update_instructions):    
    agent_pile = update_instructions[0]
    opponent_pile = update_instructions[1]

    print "update agent starting " + str(self.piles)
    if agent_pile == 0 and opponent_pile == 0:
      self.piles[0][0] = (self.piles[0][0] + self.piles[1][0]) % 5 
    elif agent_pile == 1 and opponent_pile == 0:
      self.piles[0][1] = (self.piles[0][1] + self.piles[1][0]) % 5
    elif agent_pile == 0 and opponent_pile == 1:
      self.piles[0][0] = (self.piles[0][0] + self.piles[1][1]) % 5 
    elif agent_pile == 1 and opponent_pile == 1: 
      self.piles[0][1] = (self.piles[0][1] + self.piles[1][1]) % 5
    print "update agent ending " + str(self.piles)

    return self.get_all_piles()
  
  def opponent_split_piles(self):
    total_to_split = self.piles[1][0] + self.piles[1][1]
    if total_to_split % 2 != 0:
      print "Cannot split uneven piles" 
      return self.get_all_piles()
    else: 
      self.piles[1][0] = total_to_split / 2 
      self.piles[1][1] = total_to_split / 2
      return self.get_all_piles()

  @Discontinuity_Function()
  def reset_scenario(self):
    self.piles = [[1,1],[1,1]]
    return self.piles

class Agent:
  
  def __str__(self):
    return "Agent"    
    
  def __init__(self, world):
    self._piles = StateVarRef(world.get_all_piles())
    self._world = world
    self._policies = []
    self.compute_policy()

  def compute_policy(self):
    all_piles = self._piles.get()

    action_choices = []


    if all_piles[0][0] != 0 and \
      ( all_piles[0][0] + all_piles[1][0] ) % 5 == 0:
      action_choices.append(self.add_left_left)
    elif all_piles[0][0] != 0 and \
      ( all_piles[0][0] + all_piles[1][1] ) % 5 == 0:
      action_choices.append(self.add_left_right)

    if all_piles[0][1] != 0 and \
      ( all_piles[0][1] + all_piles[1][0] ) % 5 == 0:
      action_choices.append(self.add_right_left)
    elif all_piles[0][1] != 0 and \
      ( all_piles[0][1] + all_piles[1][1] ) % 5 == 0:
      action_choices.append(self.add_right_right)

    # if I can out one of my opponent's hands 
    if len(action_choices) > 0: 
      return random.choice(action_choices)


    if ((all_piles[0][1] + all_piles[0][0]) % 2 == 0) and \
        (all_piles[0][1] != all_piles[0][0]) and \
        (all_piles[0][0] == 0 or all_piles[0][1] == 0):
      action_choices.append(self.redistribute)
    if all_piles[0][0] > 0: 
      action_choices.append(self.add_left_right)
      action_choices.append(self.add_left_left)
    elif all_piles[0][1] > 0: 
      action_choices.append(self.add_right_left)
      action_choices.append(self.add_right_right)

    print all_piles
    return random.choice(action_choices)

      
  def take_next_action(self):
    # Agent logic goes here!
    all_piles = self._piles.get()
    print all_piles

    # action = None
    # # TODO active_policy = self._policies[0]
    # active_policy = None

    action = self.compute_policy()
    # print action
    # if not action:
    #   action = self.random_move
    action()

    print "MOVE %s from %s to %s." % (action._orig_name, all_piles, self._piles)

    if self.check_win():
      print "Success at %s" % str(self._piles)
      self._piles.set(self._world.reset_scenario())
      print "World reset"
    else:  
      # ENEMY MOVE 
      enemy_funcs = []

      # left enemy hand is not empty 
      if all_piles[1][0] > 0:
        enemy_funcs += [self.opponent_left_left, self.opponent_left_right]
      # right enemy hand is not empty 
      if all_piles[1][1] > 0:
        enemy_funcs += [self.opponent_right_left, self.opponent_right_right]
      # if (all_piles[1][0] + all_piles[1][1]) % 2 == 0: 
      #   enemy_action = self.opponent_redistribute
        # enemy_funcs += [self.opponent_redistribute]
      enemy_action = random.choice(enemy_funcs)

      enemy_action()


      if self.check_fail_state():
        print "Enemy win at %s" % str(self._piles)
        self._piles.set(self._world.reset_scenario())
        print "World reset"

  def opponent_redistribute(self):
    self._piles.set(self._world.opponent_split_piles())

  # add opponent left to agent left 
  def opponent_left_left(self):
    self._piles.set(self._world.update_agent_piles((0,0)))

  # add opponent left to agent right 
  def opponent_left_right(self):
    self._piles.set(self._world.update_agent_piles((1,0)))

  # add opponent right to agent left 
  def opponent_right_left(self):
    self._piles.set(self._world.update_agent_piles((0,1)))

  # add opponent right to agent right 
  def opponent_right_right(self):
    self._piles.set(self._world.update_agent_piles((1,1)))


  @PCCA_Function('redistribute my hand')
  def redistribute(self):
    self._piles.set(self._world.split_piles(self))

  @PCCA_Function('add left to left')
  def add_left_left(self):
    self._piles.set(self._world.update_opponent_piles(self, (0,0)))

  @PCCA_Function('add left to right')
  def add_left_right(self):
    self._piles.set(self._world.update_opponent_piles(self, (0,1)))

  @PCCA_Function('add right to left')
  def add_right_left(self):
    self._piles.set(self._world.update_opponent_piles(self, (1,0)))

  @PCCA_Function('add right to right')
  def add_right_right(self):
    self._piles.set(self._world.update_opponent_piles(self, (1,1)))
  
  #@PCCA_Function('win game')
  def win_game(self):
    if self._world.achieve_goal():
      self.alert_win()
      return True
    elif self._world.take_down_one():
      self.alert_one_hand()
      return False 
    else:
      self.alert_not_win()
      return False

  def observe_world_state(self):
    self._piles.set(self._world.get_all_piles())
    
  def check_win(self):
    if self._world.check_goal():
      self.alert_win()
      return True
    else:
      return False

  def check_fail_state(self):
    if self._world.check_losing():
      self.alert_lose()
      return True
    else:
      return False

  @PCCA_Reward_Function(-10, 'lose')
  def alert_lose(self):
    return

  @PCCA_Reward_Function(-10, 'have not won')
  def alert_not_win(self):
    return

  @PCCA_Reward_Function(10, 'take down one hand')
  def alert_one_hand(self):
    return

  @PCCA_Reward_Function(10, 'win')
  def alert_win(self):
    return

def run_sim(world, steps=500):
    agent = Agent(world)
    for timestep in range(steps):
        agent.take_next_action()

    print 'Execution Finished.'


def make_world():
  world = World()
  return world

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Chopsticks Simluation.')
    parser.add_argument('-s', '--steps', dest='steps', type=int, default=500)
    args = parser.parse_args()

    world = make_world()
    run_sim(world, args.steps)