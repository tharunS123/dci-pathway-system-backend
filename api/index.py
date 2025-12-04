from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse({"message": "Hello from FastAPI on Vercel!"})

# Required by Vercel
def handler(request, context):
    return app(request, context)
