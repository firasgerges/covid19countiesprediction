import random
import math
import os
#This script generate X (default 10) folds from a dataset. this is used for X-Fold cross validation.

folds=10;
dataFile='data.csv';
shuffle=1;
trainExtension='.data';
testExtension='.test';
trainFileName='fold';
testFileName='fold';
hasHeader=0;
headerLine='';
file=open(dataFile,'r');
original=file.readlines();
processed_data=[];
testPercentage=25;
if not os.path.exists('folds'):
	os.makedirs('folds')

print('--Reading Data--');
for line in original:
	processed_data.append(line.strip('\n'));
if hasHeader:
	headerLine=processed_data[0];
	processed_data=processed_data[1:];
print('--Reading Data Complete--');

if shuffle:
	print('--Shuffeling Data--');
	random.shuffle(processed_data);
	print('--Shuffeling Data Complete--');
fold_size=math.floor(len(processed_data)/folds);
chunks=[];
index=0;
print('--Splitting Data--');
while len(chunks)<folds:
	cases=processed_data[index:index+fold_size];
	chunks.append(cases);
	index=index+fold_size;
remaining_cases=processed_data[index:];
for case in remaining_cases:
	i=random.randint(0,len(chunks)-1);
	chunks[i].append(case);
print('--Splitting Data Complete--');

print('--Generating Folds--');

for i in range(0,len(chunks)):
	test_fold=chunks[i];
	train_fold=[];
	for j in range(0,len(chunks)):
		if j!=i:
			for case in chunks[j]:
				train_fold.append(case);
	testFile=open('folds/'+testFileName+str(i)+testExtension,'w');
	trainFile=open('folds/'+trainFileName+str(i)+trainExtension,'w');
	testFile.write("\n".join(test_fold))
	trainFile.write("\n".join(train_fold))
	print ('Fold',i,'Complete')






