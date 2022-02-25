import os
from typing import List, Tuple, Optional

import joblib
from joblib import memory
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

#-----------------------------------------------
class PredictionInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    category: str


memory = joblib.Memory("cache.joblib", verbose=0)

@memory.cache(ignore=["model"])
def predict(model: Pipeline, text: str) -> int:
    return model.predict([text])[0]

#----------------------------------------------
class NewsgroupsModel:
    model: Optional[Pipeline]
    targets: Optional[List[str]]
    
    def load_model(self):
        """
        Loads the model from the file system.
        """
        model_file = os.path.join(os.path.dirname(__file__), "newsgroups_model.joblib")
        loaded_model: Tuple[Pipeline, List[str]] = joblib.load(model_file)
        model, targets = loaded_model
        self.model = model
        self.targets = targets
        
    
    async def predict(self, input: PredictionInput) -> PredictionOutput:
        """
        Runs the prediction on the text.
        """
        if not self.model or not self.targets:
            raise RuntimeError("Model not loaded.")
        prediction = predict(self.model, input.text)
        category = self.targets[prediction]
        return PredictionOutput(category=category)
    
#-------------------------------------------------------------------------
app = FastAPI()
newsgroups_model = NewsgroupsModel()

@app.post('/predict')
async def prediction_endpoint(output: PredictionOutput = Depends(newsgroups_model.predict)):
    """
    Endpoint to make a prediction.
    """
    return output    


@app.delete('/cache', status_code=204)
def delete_cache():
    """
    Deletes the cache.
    """
    memory.clear()

@app.on_event('startup')
async def startup():
    newsgroups_model.load_model()