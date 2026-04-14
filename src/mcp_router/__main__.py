"""python -m mcp_router"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("mcp_router.router:app", host="0.0.0.0", port=8765, reload=False)
