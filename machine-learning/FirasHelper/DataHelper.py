from FirasHelper.Attribute import Attribute

class DataHelper(object):

	def __init__(self):
		self.TrainD=None; #holds training data
		self.TestD=None; #holds testing data
		self.Attributes=None; #holds attributes name, type and categories
		print('Initializing a Data Helper');

	def readAttributes(self,dataPath,AttributesFileName):
		self.Attributes=[];
		attribute_file=open(dataPath+AttributesFileName,'r');
		lines=attribute_file.readlines();
		attribute_file.close();
		attribute_index=0;
		for line in lines[1:]:
			line=line.strip('\n');
			line=line.strip('.');
			attribute_instance=Attribute();
			attribute_line=line.split(':');
			attribute_instance.name=attribute_line[0].strip(' ');
			attribute_type=attribute_line[1].strip(' ');
			index=0;
			if attribute_type != 'continuous':
				attribute_instance.is_category=1;
				attribute_type=attribute_type.split(',');
				for i in range(0,len(attribute_type)):
					attribute_instance.sub_attributes.append(0);
					attribute_instance.original_options.append(attribute_type[i].strip(' '));
					attribute_instance.indices.append(index);
					index=index+1;
			else:
				attribute_instance.indices.append(index);
				index=index+1;
			self.Attributes.append(attribute_instance);

	def readCases(self,dataPath,FileName):
		print('Reading cases from: ',FileName);
		cases=[];
		data = open(dataPath+FileName, 'r');
		for line in data:
			line = line.strip('\n');
			line=line.strip('.');
			line=line.split(',');
			case=[];
			for i in range(0,len(line)-1):
				value = line[i];
				value=value.strip(' ');
				if self.Attributes[i].is_category:
					attribute_index=self.Attributes[i].original_options.index(value);
					sub_attributes=list(self.Attributes[i].sub_attributes);
					sub_attributes[attribute_index]=1;
					for sub in sub_attributes:
						case.append(int(sub));
				else:
					case.append(int(value));
			class_label_value=line[-1].strip(' ');
			class_label_value=class_label_value.strip('.');
			case.append(class_label_value);
			cases.append(case);
		return cases;

	#-------------------------------------------------------------------------------#
	#Description: Used to extract data from files and process them into training and testing data
	#
	#Arguments:
	#dataPath: path to the data directory (String)
	#TrainFileName, TestFileName: names of data files in path (String)
	#AttributesFileName: name of the attributes name file in the path. format is similar to C5.0 (String)
	#
	#Return:
	#True after successfull completion. Throw exception otherwise
	#--------------------------------------------------------------------------------#
	def registerData(self,dataPath,TrainFileName,TestFileName,AttributesFileName):
		train_data=[]
		self.readAttributes(dataPath,AttributesFileName);
		self.TrainD=self.readCases(dataPath,TrainFileName);
		self.TestD=self.readCases(dataPath,TestFileName);
		return True;
