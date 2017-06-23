######################## CSCI567 - HW4 Code ###############################
# Author: Ajay Kumar Loganathan Ravichandran
###########################################################################

###################### Deep Learning ######################################

import hw_utils 
import time, sys
import numpy as np
import time
if __name__ == "__main__":
	trainD, trainT, testD, testT = hw_utils.loaddata("MiniBooNE_PID.txt")
	trainD,testD = hw_utils.normalize(trainD, testD)
	din  = trainD.shape[1]
	dout = trainT.shape[1]

	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,dout],[din,50,dout],[din,50,50,dout],[din,50,50,50,dout]],
		'linear','softmax',[0.0],30,1000,0.001,[0.0],[0.0],False,False,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
	

	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,50,dout],[din,500,dout],[din,500,300,dout],[din,800,500,300,dout],[din,800,800,500,300,dout]],
		'linear','softmax',[0.0],30,1000,0.001,[0.0],[0.0],False,False,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
	
	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,50,dout],[din,500,dout],[din,500,300,dout],[din,800,500,300,dout],[din,800,800,500,300,dout]],
		'sigmoid','softmax',[0.0],30,1000,0.001,[0.0],[0.0],False,False,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
	
	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,50,dout],[din,500,dout],[din,500,300,dout],[din,800,500,300,dout],[din,800,800,500,300,dout]],
		'relu','softmax',[0.0],30,1000,5e-4,[0.0],[0.0],False,False,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
	
	
	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,800,500,300,dout]],
	'relu','softmax',[1e-7, 5e-7, 1e-6, 5e-6, 1e-5],30,1000, 5e-4, [0.0],[0.0],False,False,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
	
	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,800,500,300,dout]],
	'relu','softmax',[1e-7, 5e-7, 1e-6, 5e-6, 1e-5],30,1000,5e-4,[0.0],[0.0],False,True,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
	
	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,800,500,300,dout]],
	'relu','softmax', [5e-7],100,1000,1e-5,[1e-5, 5e-5, 1e-4, 3e-4, 7e-4, 1e-3],[0.0],False,False,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
	
	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,800,500,300,dout]],
	'relu','softmax', [0.0],50,1000,1e-5,[1e-5],[0.99,0.98,0.95,0.9,0.85],True,False,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
	
	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,800,500,300,dout]],
	'relu','softmax', [1e-7],100,1000,1e-5,[1e-5],[0.99],True,True,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
	
	start = time.time()
	hw_utils.testmodels(trainD,trainT,testD,testT,[[din,50,dout],[din,500,dout],[din,500,300,dout],[din,800,500,300,dout],[din,800,800,500,300,dout]],
	'relu','softmax', [1e-7,5e-7,1e-6,5e-6,1e-5],100,1000,1e-5,[1e-5,5e-5,1e-4],[0.99],True,True,0)
	end = time.time()
	print 'Time elapsed :' +str(end - start)
