from fastapi import FastAPI, Form, Body
from fastapi.responses import FileResponse
from ml.ml import load_model
 
app = FastAPI()
 
@app.get("/")
def root():
    return FileResponse("public/index.html")
 
 
@app.post("/hello")
def hello(data = Body()):
    model_predict = load_model(data["ads"], 
    	data["add"], data["adp"], data["zdin"],
    	data["zdout"], data["zel"], data["weight"],
    	data["accom"], data["hear"], data["balance"])
    return {"message": f" ваш bioage - {model_predict}"}
