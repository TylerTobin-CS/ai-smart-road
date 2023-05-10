# Tyler Tobin - 001105522 - FYP 1682
# Road situation classifier using trained model
# Must be run after server_stack.py

#----------- Imports ----------

import pickle
import sklearn
import csv
import socket
import select
import errno
import sys

#------------ Network config ----------

HEADER_LENGTH = 10
IP = "192.168.0.136"
PORT = 12345

my_username = "classifier"
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))
client_socket.setblocking(False)

username = my_username.encode("utf-8")
username_header = f"{len(username):<{HEADER_LENGTH}}".encode("utf-8")
client_socket.send(username_header + username)

print("connected to server")

#--------- Model loading using Pickle ----------

loaded_model = pickle.load(open("road_situation_classifier_model_1.pkl", "rb"))
print("Model loaded")

#---------- Model function ---------

def situationClassifier(model):
    with open("aiChain.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            print(row)
            x = [[int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7])]]
            print(x)
            prediction = model.predict(x)
            message = prediction[0]
            message = message.encode("utf-8")
            message_header = f"{len(message):<{HEADER_LENGTH}}".encode("utf-8")
            client_socket.send(message_header + message)

#---------- Main loop ----------
            
while True:
    
    with open("aiChain.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        if not any(reader):
            continue
        else:
            situationClassifier(loaded_model)
            with open("aiChain.csv", mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([])
            csvfile.close()

#----------- Test predictions ----------

'''
prediction = loaded_model.predict([[1, 0, 1, 1, 0, 0, 1589, 0]])
prediction2 = loaded_model.predict([[1, 1, 0, 0, 1, 0, 1620, 2187]])
print(prediction[0])
print(prediction2[0])
'''
