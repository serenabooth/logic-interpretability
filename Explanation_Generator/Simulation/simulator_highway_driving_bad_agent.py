# -*- coding: utf-8 -*-
"""
Simulator for Chopsticks game example
"""

import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Code/'))
sys.path.insert(0,lib_path)

import copy
import random
import tornado.web
import tornado.websocket
import tornado.ioloop
import argparse
from pcca import *
from pcca_decorator import *
import random 
global global_pcca

class World:
  
  def __init__(self):
    #[[left],[front, me, behind],[right],[exit#]]; 0 empty, 1 obstacle, 2 me, 41 - before my exit, 42 - my exit 
    self.state = [[random.randint(0,1)],[random.randint(0,1),2,random.randint(0,1)],[random.randint(0,1)],[random.randint(41,42)]]
    self.create_predicate_functions()

  def create_predicate_functions(self):
    @Planning_Predicate("a vehicle is to my left")
    def car_left(state):
      if (state[0][0][0]) == 1:
        return True
      return False

    @Planning_Predicate("a vehicle is to my right")
    def car_right(state):
      if (state[0][2][0]) == 1:
        return True
      return False

    @Planning_Predicate("a vehicle is in front of me")
    def car_ahead(state):
      if (state[0][1][0]) == 1:
        return True
      return False

    @Planning_Predicate("a vehicle is behind me")
    def car_behind(state):
      if (state[0][1][2]) == 1:
        return True
      return False

    @Planning_Predicate("the next exit is 42")
    def next_exit(state):
      if (state[0][3][0]) == 42:
        return True
      return False


  def get_current_state(self):
    return self.state
    
  def achieve_goal(self):
    if self.check_goal():
      return True
    return False

  def check_goal(self):
    """
    Check if didn't crash
    """
    return True

  def check_losing(self):
    """
    Check if crashed
    """
    return False

  def speed_up(self, agent):
    if self.state[1][0] == 1:
      print "cannot speed up"
      return self.state
    else: 
      proposed_state = copy.deepcopy(self.state)
      proposed_state[1][0] = 2
      proposed_state[1][1] = 0
      self.state = proposed_state
      return proposed_state

  def slow_down(self, agent):
    if self.state[1][2] == 1:
      print "cannot slow down"
      return self.state
    else: 
      proposed_state = copy.deepcopy(self.state)
      proposed_state[1][2] = 2
      proposed_state[1][1] = 0
      self.state = proposed_state
      return proposed_state

  def merge_left(self, agent):
    if self.state[0][0] == 1:
      print "cannot merge left"
      return self.state
    else: 
      proposed_state = copy.deepcopy(self.state)
      proposed_state[0][0] = 2
      proposed_state[1][1] = 0
      self.state = proposed_state
      return proposed_state

  def merge_right(self, agent):
    if self.state[2][0] == 1:
      print "cannot merge left"
      return self.state
    else: 
      proposed_state = copy.deepcopy(self.state)
      proposed_state[2][0] = 2
      proposed_state[1][1] = 0
      self.state = proposed_state
      return proposed_state

  def do_nothing(self, agent):
    return self.state

  @Discontinuity_Function()
  def reset_scenario(self):
    self.state = [[random.randint(0,2)],[random.randint(0,2),2,random.randint(0,2)],[random.randint(0,2)],[random.randint(41,43)]]
    return self.state

class Agent:
  
  def __str__(self):
    return "Agent"    
    
  def __init__(self, world):
    self._state = StateVarRef(world.get_current_state())
    self._world = world
    self._policies = []
    self.compute_policy()

  def compute_policy(self):
    state = self._state.get()

    car_left = state[0][0]
    car_right = state[2][0]
    car_front = state[1][0]
    car_behind = state[1][2]
    car_self = state[1][1]
    exit_sign = state[3][0]

    # if it's not my exit and no one is in front of me, speed up
    # or, if it's my exit, someone is behind me, and someone is to my right
    if (exit_sign == 42 and car_front == 1 and car_right == 1 and car_behind == 1) or \
       (exit_sign != 42 and car_front == 1):
      return self.increase_speed
    # if it's not my exit and someone is in front of me and no one is left of me, merge left 
    # or if it's not my exit and there's someone behind me and there's no one to my left
    elif (exit_sign != 42 and car_left == 1 and car_front == 1) or \
       (exit_sign != 42 and car_left == 1 and car_behind == 1):
      return self.merge_left
    # if it's my exit and no one is to the right of me, merge right
    # or, if it's not my exit and someone is to my left and someone is in front of me, merge right
    elif (exit_sign == 42 and car_right == 1) or\
       (exit_sign != 42 and car_right == 1 and car_left == 1 and car_front == 1):
      return self.merge_right
    # # if it's my exit and someone is to the right of me, slow down 
    # elif (exit_sign == 42 and car_right == 1) or \
    #    (exit_sign != 42 and car_front == 1 and car_behind == 0):
    #   return self.slow_down
    else:
      return self.do_nothing
      
  def take_next_action(self):
    state = self._state.get()

    action = self.compute_policy()
    # print action
    # if not action:
    #   action = self.random_move
    action()

    print "MOVE %s from %s to %s." % (action._orig_name, state, self._state)
    # check win state
    if self.check_win():
      print "Success at %s" % str(self._state)
      self._state.set(self._world.reset_scenario())
      print "World reset"
    # check lose state
    if self.check_fail_state():
      print "Lose at %s" % str(self._state)
      self._state.set(self._world.reset_scenario())
      print "World reset"

  @PCCA_Function('slow down')
  def slow_down(self):
      self._state.set(self._world.slow_down(self))

  @PCCA_Function('speed up')
  def increase_speed(self):
      self._state.set(self._world.speed_up(self))

  @PCCA_Function('merge left')
  def merge_left(self):
      self._state.set(self._world.merge_left(self))

  @PCCA_Function('merge right')
  def merge_right(self):
      self._state.set(self._world.merge_right(self))

  @PCCA_Function('do nothing')
  def do_nothing(self):
      self._state.set(self._world.do_nothing(self))


  def observe_world_state(self):
    self._state.set(self._world.get_current_state())
    
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
    parser = argparse.ArgumentParser(description='Run Highway Simluation.')
    parser.add_argument('-s', '--steps', dest='steps', type=int, default=500)
    args = parser.parse_args()

    world = make_world()
    run_sim(world, args.steps)