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
    #[alert_and_able_to_move, breathing, open_airway, respiratory_rate, capillary_refill, pulse_rate]
    self.state = [0, random.randint(0,1), random.randint(0,1), random.randint(0, 1), random.randint(0,1), random.randint(0,1)]
    self.create_predicate_functions()

  def patient_state_dict(self, state):
    patient_state = {'alert_and_able_to_move': state[0], 
                     'breathing': state[1], 
                     'open_airway': state[2],
                     'respiration_rate': state[3],
                     'capillary_refill': state[4],
                     'pulse_rate': state[5]}
    return patient_state

  def create_predicate_functions(self):
    @Planning_Predicate("patient is alert and able to move")
    def alert_able_to_move(state):
      patient_state_dict = self.patient_state_dict(state[0])
      if (patient_state_dict['alert_and_able_to_move']) == 1:
        return True
      return False

    @Planning_Predicate("patient is breathing")
    def can_breathe(state):
      patient_state_dict = self.patient_state_dict(state[0])
      if (patient_state_dict['breathing']) == 1:
        return True
      return False

    @Planning_Predicate("patient is able to breathe after opening airway")
    def open_airway(state):
      patient_state_dict = self.patient_state_dict(state[0])
      if (patient_state_dict['open_airway']) == 1:
        return True
      return False

    @Planning_Predicate("respiration count is between 10 and 30")
    def respiration_rate(state):
      patient_state_dict = self.patient_state_dict(state[0])
      if patient_state_dict['respiration_rate'] == 1:
        return True
      return False

    @Planning_Predicate("capillary refill takes less than 2 seconds")
    def capillary_refill(state):
      patient_state_dict = self.patient_state_dict(state[0])
      if patient_state_dict['capillary_refill'] == 1:
        return True
      return False

    @Planning_Predicate("pulse is between 40 and 70")
    def pulse_rate(state):
      patient_state_dict = self.patient_state_dict(state[0])
      if patient_state_dict['pulse_rate'] == 1:
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

  def action(self, agent):
      return self.state

  @Discontinuity_Function()
  def reset_scenario(self):
    #[alert_and_able_to_move, breathing, open_airway, respiratory_rate, capillary_refill, pulse_rate]
    self.state = [0, random.randint(0,1), random.randint(0,1), random.randint(0, 1), random.randint(0,1), random.randint(0,1)]
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
    patient_state = self._world.patient_state_dict(state)
    # patient_state = {'alert_and_able_to_move': state[0], 
    #                  'breathing': state[1], 
    #                  'open_airway': state[2],
    #                  'respiration_rate': state[3],
    #                  'capillary_refill': state[4],
    #                  'pulse_rate': state[5]}


    print (state)
    if patient_state['alert_and_able_to_move']:
      return self.delayed

    if not patient_state['breathing'] and patient_state['open_airway']:
      return self.immediate
    
    elif not patient_state['breathing'] and not patient_state['open_airway']:
      return self.expectant
    
    if patient_state['breathing']:
      if not patient_state['respiration_rate']:
        return self.immediate
      elif patient_state['capillary_refill']:
        return self.urgent
      elif not patient_state['capillary_refill']:
        if not patient_state['pulse_rate']:
          return self.immediate
        else: 
          return self.urgent
 
  def take_next_action(self):
    state = self._state.get()

    action = self.compute_policy()
    if action: 
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

  @PCCA_Function('triage: immediate')
  def immediate(self):
      self._state.set(self._world.action(self))

  @PCCA_Function('triage: urgent')
  def urgent(self):
      self._state.set(self._world.action(self))

  @PCCA_Function('triage: delayed')
  def delayed(self):
      self._state.set(self._world.action(self))

  @PCCA_Function('triage: expectant')
  def expectant(self):
      self._state.set(self._world.action(self))

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