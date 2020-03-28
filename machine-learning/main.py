from FirasHelper.DataHelper import DataHelper
from FirasHelper.Classifier import Classifier
import sys
data_path='Data/';
train_name='folds/fold';
test_name='folds/fold';
train_extension='.data';
test_extension='.test';
attribute_name='data.name';
result_name='Results/result';
result_extension='.csv';
first_fold=0;
last_fold=9;
runs=[
	{'method':'LR', 'arguments':{},'id':1},
];

for i in range(0,len(runs)):
	run=runs[i];
	id=str(run['id']);
	writer=open(result_name+'_'+run['method']+'_'+str(run['id'])+result_extension,'w');
	for f in range(first_fold,last_fold+1):
		print('Fold: ',f);
		train_file=train_name+str(f)+train_extension;
		test_file=test_name+str(f)+test_extension;
		data_helper= DataHelper();
		data_helper.registerData(data_path,train_file,test_file,attribute_name);
		train_data=data_helper.TrainD;
		test_data=data_helper.TestD;
		attributes=data_helper.Attributes;
		print('------Data Registered-------');
		classifier=Classifier();
		classifier.runClassifier(run['method'],train_data,test_data,run['arguments']);
		result={'train':train_file,'test':test_file,'method':run['method'],'arguments':classifier.Arguments};
		id_fold=run['method']+"_fold_"+str(f);
		result['trainAcc']=classifier.computeRMSE('TRAIN',id_fold);
		result['testAcc']=classifier.computeRMSE('TEST',id_fold);
		print(result['testAcc'])
		line=str(result['train'])+','+str(result['test'])+','+str(result['method'])+','+str(id)+','+str(result['trainAcc'])+','+str(result['testAcc']);
		writer.write(line+'\n');
	writer.close();