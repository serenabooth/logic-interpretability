import json
from websocket import create_connection

ws = create_connection("ws://127.0.0.1:1235/")

print ("\nWhat do you do?")
ws.send(json.dumps("ask_question What do you do?"))
result =  ws.recv()
print (result)

print ("\nWhen do you redistribute your hand?")
ws.send(json.dumps("ask_question When do you redistribute your hand?"))
result =  ws.recv()
print (result)

print ("\nWhen do you add left to left?")
ws.send(json.dumps("ask_question When do you add left to left?"))
result =  ws.recv()
print (result)

print ("\nWhen do you add left to right?")
ws.send(json.dumps("ask_question When do you add left to right?"))
result =  ws.recv()
print (result)

print ("\nWhen do you add right to left?")
ws.send(json.dumps("ask_question When do you add right to left?"))
result =  ws.recv()
print (result)

print ("\nWhen do you add right to right?")
ws.send(json.dumps("ask_question When do you add right to right?"))
result =  ws.recv()
print (result)

ws.close()