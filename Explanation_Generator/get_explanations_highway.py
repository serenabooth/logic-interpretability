import json
from websocket import create_connection

ws = create_connection("ws://127.0.0.1:1235/")

print ("\nWhat do you do?")
ws.send(json.dumps("ask_question What do you do?"))
result =  ws.recv()
print (result)

print ("\nWhen do you slow down?")
ws.send(json.dumps("ask_question When do you slow down?"))
result =  ws.recv()
print (result)

print ("\nWhen do you speed up?")
ws.send(json.dumps("ask_question When do you speed up?"))
result =  ws.recv()
print (result)

print ("\nWhen do you merge left?")
ws.send(json.dumps("ask_question When do you merge left?"))
result =  ws.recv()
print (result)

print ("\nWhen do you merge right?")
ws.send(json.dumps("ask_question When do you merge right?"))
result =  ws.recv()
print (result)

print ("\nWhen do you do nothing?")
ws.send(json.dumps("ask_question When do you do nothing?"))
result =  ws.recv()
print (result)

ws.close()