import json
from websocket import create_connection

ws = create_connection("ws://127.0.0.1:1235/")

print ("\nWhat do you do?")
ws.send(json.dumps("ask_question What do you do?"))
result =  ws.recv()
print (result)

print ("\nWhen do you triage: immediate?")
ws.send(json.dumps("ask_question When do you triage: immediate?"))
result =  ws.recv()
print (result)

print ("\nWhen do you triage: delayed?")
ws.send(json.dumps("ask_question When do you triage: delayed?"))
result =  ws.recv()
print (result)

print ("\nWhen do you triage: urgent?")
ws.send(json.dumps("ask_question When do you triage: urgent?"))
result =  ws.recv()
print (result)

ws.close()