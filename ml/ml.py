import pickle
import numpy as np

filename_man = 'ml/rf_man_40.pkl'
filename_women = 'ml/rf_women_40.pkl'

with open(filename_man, 'rb') as file:
    model_men = pickle.load(file)

with open(filename_women, 'rb') as file:
    model_women = pickle.load(file)
    
def load_model_men(ADS: int,
		ADD: int,
		ADP: int,
		ZDin: int,
		ZDout: int,
		ZEL: int,
		Weight: int,
		Accommodation: int,
		Hear: int,
		Balance:int) -> int:
		
	""" ADS - systolic pressure,
	 ADD - dystolic pressure,
	 ADP - difference between systolitic and dystolitic pressure,
	 ZDin - holding your breath while inhaling in sec
	 ZDout - breath retention on exhalation in seconds
	 ZEL - vital capacity of the lungs in ml
	 Weight - weight of your body, kg
	 Accommodation - accommodation in diopters
	 Hear - hearing acuity in belarus
	 Balance - static balancing in seconds
	 """
	 
	x = np.array([ADS, ADD, ADP, ZDin, ZDout, ZEL, Weight, Accommodation, Hear, Balance]).reshape(1,10)
	res = model_men.predict(x)
	return np.round(res[0])

def load_model_women(ADS: int,
		ADD: int,
		ADP: int,
		ZDin: int,
		ZDout: int,
		ZEL: int,
		Weight: int,
		Accommodation: int,
		Hear: int,
		Balance:int) -> int:
		
	""" ADS - systolic pressure,
	 ADD - dystolic pressure,
	 ADP - difference between systolitic and dystolitic pressure,
	 ZDin - holding your breath while inhaling in sec
	 ZDout - breath retention on exhalation in seconds
	 ZEL - vital capacity of the lungs in ml
	 Weight - weight of your body, kg
	 Accommodation - accommodation in diopters
	 Hear - hearing acuity in belarus
	 Balance - static balancing in seconds
	 """
	 
	x = np.array([ADS, ADD, ADP, ZDin, ZDout, ZEL, Weight, Accommodation, Hear, Balance]).reshape(1,10)
	res = model_women.predict(x)
	return np.round(res[0])	
		
		
