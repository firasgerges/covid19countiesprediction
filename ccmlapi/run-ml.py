import sys
import pickle
if len(sys.argv)!=14:
	sys.exit("Invalid Arguments");
density=sys.argv[1];
education=sys.argv[2];
unemployment=sys.argv[3];
sex_ratio=sys.argv[4];
age_median=sys.argv[5];
public_commute=sys.argv[6];
infection_rate=sys.argv[7];
infection_density=sys.argv[8];
average_increase=sys.argv[9];
covid1=sys.argv[10];
covid2=sys.argv[11];
covid3=sys.argv[12];
covid4=sys.argv[13];

temp=[density, education, unemployment, sex_ratio, age_median, public_commute, infection_rate, infection_density, average_increase, covid1, covid2, covid3, covid4];
case=[];
for i in temp:
	case.append(float(i));

print(case)

loaded_model = pickle.load(open("ml_model.sav", 'rb'))
result = loaded_model.predict([case])[0]
print(int(result));


