"""
Main FastAPI Application

This module sets up the main FastAPI application, including the root
endpoint and the inclusion of the inference API router.
"""

import uvicorn
from fastapi import FastAPI

from routers.infapi import infapi

app = FastAPI()


@app.get("/")
async def root() -> str:
    """
    Root API endpoint

    Returns
    -------
    str
        A welcome message.
    """
    return "Welcome!"


app.include_router(infapi)

if __name__ == "__main__":
    config = uvicorn.Config(
        "main:app", host="0.0.0.0", reload=True, port=8080, log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
