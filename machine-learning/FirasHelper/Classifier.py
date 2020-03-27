from sklearn import svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import math
import numpy as np
class Classifier(object):

	def __init__(self):
		self.TrainD=None; #holds training data
		self.TestD=None; #holds testing data
		self.ml=None; #holds type of classifier to use
		self.TrainPrediction=None; #holds predicted class label of training set
		self.TestPrediction=None; # holds predicted class label of testing set
		self.TrainActual=None; #holds real class label of training set
		self.TestActual=None; #holds real class label of testing set
		self.Arguments=None;
		print('Initializing a Classifier Object');

	def runClassifier(self,ml,TrainD,TestD,Arguments):
		self.TrainD=TrainD;
		self.TestD=TestD;
		self.ml=ml;
		self.TrainPrediction=[];
		self.TestPrediction=[];
		self.TrainActual=[];
		self.TestActual=[];
		self.Arguments=Arguments;
		Tr=[];
		Te=[];

		for case in self.TrainD:
			self.TrainActual.append(case[-1]);
			Tr.append(case[0:len(case)-1]);
		for case in self.TestD:
			self.TestActual.append(case[-1]);
			Te.append(case[0:len(case)-1]);
		TrC=self.TrainActual;
		TeC=self.TestActual;
		clf=None;
		print(ml,Arguments,'status: ', 'Fitting Started');
		if self.ml=='RF':
			clf=RandomForestRegressor(n_estimators=100)
		if self.ml=='DT':
			clf = DecisionTreeRegressor()
		if self.ml=='LR':
			clf = LinearRegression()
		if self.ml=='GBM':
			clf = GradientBoostingRegressor()
		if self.ml=='SVR':
			clf =SVR(gamma='scale', C=1.0, epsilon=0.2)
		clf.fit(Tr,TrC);
		print(ml,Arguments,'status: ', 'Fitting Done');
		print(ml,Arguments,'status: ', 'Training Set Prediction Started');
		for i in Tr:
			self.TrainPrediction.append(clf.predict([i])[0])
		print(ml,Arguments,'status: ', 'Training Set Prediction Done');
		print(ml,Arguments,'status: ', 'Testing Set Prediction Started');
		for i in Te:
			self.TestPrediction.append(clf.predict([i])[0]);
		print(ml,Arguments,'status: ', 'Testing Set Prediction Done');

		return True;

	def computeRMSE(self,type,id='None'):
		actual=None;
		predicted=None;
		if type=='TRAIN':
			actual=self.TrainActual;
			predicted=self.TrainPrediction
		if type=='TEST':
			actual=self.TestActual;
			predicted=self.TestPrediction
		writer=open(type+"_"+str(id)+".csv",'w');
		for i in range(0,len(actual)):
			writer.write(str(float(actual[i]))+","+str(float(predicted[i])));
			writer.write("\n");
		RMSE=0;
		for i in range(0,len(actual)):
			RMSE = RMSE+ pow((float(actual[i]) - float(predicted[i])),2)
		RMSE=float(RMSE)/float(len(actual));
		RMSE=math.sqrt(RMSE)
		return RMSE;

