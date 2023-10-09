from fastapi import FastAPI, Form, Body
from fastapi.responses import FileResponse
from ml.ml import load_model_men, load_model_women
 
app = FastAPI()
 
@app.get("/")
def root():
    return FileResponse("public/index.html")
 
 
@app.post("/hello")
def hello(data = Body()):
    if data['sex'] == 'man':
        model_predict = load_model_men(data["ads"], 
    	data["add"], data["adp"], data["zdin"],
    	data["zdout"], data["zel"], data["weight"],
    	data["accom"], data["hear"], data["balance"])
    else:
    	model_predict = load_model_women(data["ads"], 
    	data["add"], data["adp"], data["zdin"],
    	data["zdout"], data["zel"], data["weight"],
    	data["accom"], data["hear"], data["balance"])
    return {"message": f" ваш bioage - {model_predict}"}
