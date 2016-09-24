import numpy as np
def convolution(X,F,S,P,W,B):
	''' Convolution operation on square 3D signals, X: Signal, F:Filter, S: Stride, P padding to get desired output size,
	W:Weight matrix, B: Bias value'''
	X=np.pad(X,[(P,P),(P,P),(0,0)],'constant',constant_values=0) # pad edges with zeros 
	a=X.shape[0]
	b=F[0]
	c=((a-b)/S)+1 #output Dimensions
	v=a*1.0
	n=b
	m=((v-n)/S)+1
	if((m).is_integer() !=True or X.shape[2] != F[2]): 
		print "Invalid Inputs in Conv Layer"
		return 0,0 # return Zero and raise error : incompatible convolution operands
	else:
		out = np.repeat((np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1)+ np.repeat(xrange(0,b,1),b).reshape(1,-1)),c,axis=0) #obtain row indicies of the matrix
		out1=np.tile(np.repeat(np.repeat(xrange(0,b,1),1).reshape(1,-1)+np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1),b,axis=0),(c,1))#Obtain column indicies of the matrix
		im2col=X[out.flatten(),out1.flatten()].reshape(1,c*c*b*b*X.shape[2]).reshape(b*b*X.shape[2],c*c,order="F")
		output_volume=np.dot(W,im2col).reshape(c,c,W.shape[0])+B
		return output_volume,im2col
		
		
def maxpool(X,F,S):
	a=X.shape[0]
	b=F[0]
	c=((a-b)/S)+1
	v=a*1.0
	n=b
	m=((v-n)/S)+1
	if((m).is_integer() !=True or X.shape[2] != F[2]): 
		print "Invalid Inputs in Pool Layer"
		return 0
	l=np.zeros((b*b,c*c*X.shape[2]))
	out = np.repeat((np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1)+ np.repeat(xrange(0,b,1),b).reshape(1,-1)),c,axis=0)#row indicies
	out1=np.tile(np.repeat(np.repeat(xrange(0,b,1),1).reshape(1,-1)+np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1),b,axis=0),(c,1))#column indicies
	im2col=X[out.flatten(),out1.flatten()].reshape(1,c*c*b*b*X.shape[2],order='C').reshape(c*c*b*b,X.shape[2],order='F')
	out_volume=np.amax(im2col,axis=0).reshape(c,c,X.shape[2])
	return out_volume
def col2im(im2col,P,F,X,S):
	X[0]=X[0]+2*P
	X[1]=X[1]+2*P
	a=X[0]
	b=F[0]
	c=((a-b)/S)+1#output Dimensions
	im2col=im2col.reshape(1,c*c*b*b*X[2],order="F").reshape(c*c*b*b,X[2],order="C")
	out = np.repeat((np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1)+ np.repeat(xrange(0,b,1),b).reshape(1,-1)),c,axis=0)#row indicies
	out1=np.tile(np.repeat(np.repeat(xrange(0,b,1),1).reshape(1,-1)+np.repeat(xrange(0,a-b+1,S),1).reshape(-1,1),b,axis=0),(c,1))#column indicies
	Xa=np.empty(((X[0],X[1],X[2])))
	Xa[out.flatten(),out1.flatten()]=im2col[xrange(im2col.shape[0])]
	return Xa[P:X[0]-P,P:X[0]-P]
	
