from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"message": "Don is building a production ML API!"}

