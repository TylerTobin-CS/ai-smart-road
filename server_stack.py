# Tyler Tobin - 001105522 - FYP 1682
# Server stacking program to create an input to the classifer
# Must be run after server.py

#--------- Imports ---------

import csv
import os
from time import sleep

#---------- Stack function ---------

def stack():
    
    stack_timer = 0
    element0 = 0 # East camera
    element1 = 0 # West camera
    element2 = 0 # Obstruction
    element3 = 0 # Number of vehicles == 1
    element4 = 0 # Number of vehicles == 2
    element5 = 0 # Number of vehicles == 2
    element6 = 0 # East weight
    element7 = 0 # West weight
    weight_east = 0
    weight_west = 0
    no_of_vehicles_east = 0
    no_of_vehicles_west = 0
    stack_timer = 0
    while stack_timer < 3:
        no_of_vehicles = 0
        element6 = 0
        element7 = 0
        stack_timer = stack_timer + 1
        sleep(1)
        with open("carDetected.csv", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                if len(row) == 0:
                    print("row is showing nothing")
                    continue
                if row[0] == "cam_east":
                    element0 = 1
                    no_of_vehicles = no_of_vehicles + 1
                    element6 = element6 + int(row[1])
                    
                elif row[0] == "cam_west":
                    element1 = 1
                    no_of_vehicles = no_of_vehicles + 1
                    element7 = element7 + int(row[1])
                elif row[0] == "cam_central":
                    element2 = 1
                
            csvfile.close()  
            
            if no_of_vehicles == 1:
                element3 = 1
            elif no_of_vehicles == 2:
                element4 = 1
                element3 = 0
            else:
                element5 = 1
                element3 = 0
                element4 = 0
            
    MLinput = [element0, element1, element2, element3, element4, element5, element6, element7]
    element0 = 0 # East camera
    element1 = 0 # West camera
    element2 = 0 # Obstruction
    element3 = 0 # Number of vehicles == 1
    element4 = 0 # Number of vehicles == 2
    element5 = 0 # Number of vehicles == 2
    element6 = 0 # East weight
    element7 = 0 # West weight
    with open("carDetected.csv", mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([])
    #csvfile.close()
    
    #save to a new csv for saving the ai chain
    with open("aiChain.csv", mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(MLinput)
    csvfile.close()
    
    print(MLinput)
    return MLinput

#--------- Main loop ----------

while True:
    
    with open("carDetected.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        if not any(reader):
            continue
        else:
            print("Stack in progress")
            stack()
