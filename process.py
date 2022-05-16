
import os, signal
  
def process():
     
    # Ask user for the name of process
    name = input("Enter process Name: ")
    try:
         
        # iterating through each instance of the process
        for line in os.popen("ps -ef | grep " + name + " | grep -v grep"):
            fields = line.split()
             
            # extracting Process ID from the output
            pid = fields[2]
            print('ID: ', pid)
            # terminating process
            os.kill(int(pid),"fully terminated")
         
    except:
        print("Error Encountered while running script")
  
process()