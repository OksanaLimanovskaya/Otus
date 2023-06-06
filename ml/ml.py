import pickle
import numpy as np

filename = 'ml/model_RF.pkl'

with open(filename, 'rb') as file:
    model = pickle.load(file)
    
def load_model(ADS: int,
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
	res = model.predict(x)
	return np.round(res[0])
	
		
		
