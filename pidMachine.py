'''
PID Controller
Motor: AndyMark
Motor Driver: Basic Micro Roboclaw
Author: Prabin Rath 
'''

import time
from roboclaw import Roboclaw
import socket

host = socket.gethostname()  # get local machine name
port = 1234  # Make sure it's within the > 1024 $$ <65535 range
s = socket.socket()
s.bind((host, port))
s.listen(1)
c, addr = s.accept()
print("Connection from: " + str(addr))

rc = Roboclaw("COM7",115200)
rc.Open()
address = 0x80
counter=1

def getDataArray(encoded):
	#encoded=encoded.decode() #for Python 3.6
	dat=encoded.split('#')
	#print(dat)
	vals=[]
	for temp in dat:
		if len(temp)>0:
			try:
				vals.append(float(temp))
			except:
				pass
	return vals

def mapSpeed(spd):
	#This function has values from the roboclaw motor driver documentation. max PWM = 32767
	#This function has experimentally noted values
	#Note: Do not change it
	if spd<-30767:
		spd=-30767
	elif spd>30767:
		spd=30767
	if spd<0:
		spd-=2000
	else:
		spd+=2000
	return int(spd)

def mapData(pos):
	#This function has values from the roboclaw motor driver documentation. steps per turn = 994
	#This function has experimentally noted values
	#Note: Do not change it
	return int((float(pos)/360)*1994)

def takeAction(params=[]):
	global counter
	setPoint=params[0]
	Kp=params[1];Ki=params[2];Kd=params[3];
	#rc.SetEncM1(address,0) #to set absolute or relative mode of rotation
	setPoint=mapData(setPoint)
	p_e=0;i_e=0;d_e=0;e_prev = 0;e_curr = 0;
	temp=rc.ReadEncM1(address)[1]
	ovL=temp;ovU=temp;
	init_time=time.clock()
	while True:
		cur=rc.ReadEncM1(address)[1]
		if cur<ovL:
			ovL=cur
		if cur>ovU:
			ovU=cur
		if cur>setPoint-2 and cur<setPoint+2:
			print 'Reached Target :) Iteration Number: ',counter
			counter+=1
			rc.DutyM1(address,0)
			break
		e_curr = setPoint - cur
		i_e = p_e + e_curr
		if i_e>100 or i_e<-100:
				i_e=0
		p_e = e_curr
		d_e = e_curr - e_prev
		u = Kp * p_e + Ki * i_e + Kd * d_e
		e_prev=e_curr
		print p_e,' ',i_e,' ',d_e,' ',u,' ',mapSpeed(u) #verbose output for debugging
		rc.DutyM1(address,mapSpeed(u))
		time.sleep(0.01) #sampling frequency: Determines the overshoot greatly
	response_time=time.clock()-init_time
	overshoot = 100
	if ovL==temp:
		overshoot=ovU-setPoint
	elif ovU==temp:
		overshoot=setPoint-ovL
	if overshoot==-1 or overshoot==1:
		overshoot=0
	return [overshoot,response_time]

while True:
	data = c.recv(1024)
	#if not data:
	#	break
	try:
		res=takeAction(getDataArray(data))
		print res,"\n\n"
		#time.sleep(1)
		to_send=str(res[0])+'#'+str(res[1])
	except:
		to_send='50#50'
	c.send(to_send)

c.close()

#print takeAction([90,60.0,0.016610932,159.69377])