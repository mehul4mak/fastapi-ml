"""Main FastAPI Application """

import uvicorn
from fastapi import FastAPI

from routers.infapi import infapi

app = FastAPI()


@app.get("/")
async def root():
    """Root api"""
    return "Welcome!"


app.include_router(infapi)


if __name__ == "__main__":
    config = uvicorn.Config(
        "main:app", host="0.0.0.0", reload=True, port=8080, log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
