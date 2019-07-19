# -*- coding: utf-8 -*-
import copy
import dill as pickle
import json
import tornado.web
import tornado.websocket
import tornado.ioloop

import sys, os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Simulation/'))
sys.path.insert(0,lib_path)

from simulator_highway_driving import *
from simulator_chopsticks import *
from pcca import *


class PCCAServer(object):
  '''
    This class exposes a websocket interface for asking questions, getting answers, and exporting graph json from an internal PCCA model
  '''
  
  class PCCAServerHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
      self._pcca = kwargs.pop('pcca')
      self._simulator = kwargs.pop('simulator')
      super(PCCAServer.PCCAServerHandler,self).__init__(*args,**kwargs)
    
    def open(self):      
      print "Connection established."
  
    def check_origin(self, origin):
      return True
  
    def on_message(self, message):
      message = json.loads(message)

      if self._simulator:
        print "\n\n\n\n\n\n\n\n\n"
        print self._simulator.get_state()

      print "Received message: " + message
      if ' ' not in message:
        cmd = message
        arg = ''
      else:
        cmd = message[:message.index(' ')]
        arg = message[len(cmd):]
        
      print "Command: %s, Argument: %s" % (cmd, arg)
      # Inbound message handling
      if cmd == 'exit':
        tornado.ioloop.IOLoop.instance().stop()
        print "Server stopped."
        exit()
      elif cmd == 'get_state':
        response_msg = "Error - World state request sent to PCCAServer instead of SimulationServer!"
      elif cmd == 'ask_question':
        response_msg = 'answer_question ' + str(self._pcca.answer_question(arg))
      else:
        response_msg = 'Error - Invalid command received!'
        
      print "Sending back: " + response_msg
      self.write_message(response_msg)
  
    def on_close(self):
      print "Connection closed."


  def __init__(self, port, pcca=None, simulator=None):
    self._port = port
    self._pcca = None
    self._simulator = simulator
    
    if type(pcca) is str:
      try:
        self._pcca = PCCA.load_from_file(pcca)
      except:
        print "Could not open file %s, %s" % (pcca, str(sys.exc_info()[0]))
        raise Exception
    elif type(pcca) is PCCA:
      self._pcca = pcca
    else:
      print "Invalid PCCA argument in PCCAServer construction."
      raise Exception('Could not initialize PCCA Server')

    self._socket = tornado.web.Application([(r"/", PCCAServer.PCCAServerHandler, {'pcca': self._pcca, 'simulator': self._simulator})])

      
  def start_server(self):
    '''
      Starts a websocket socket on port self._port
    '''
    self._socket.listen(self._port)
    tornado.ioloop.IOLoop.instance().start()  

if __name__ == '__main__':
  server = PCCAServer(1235, '../Simulation/Chopsticks_Bad_Checkpoints/checkpoint-15000')
  server.start_server()