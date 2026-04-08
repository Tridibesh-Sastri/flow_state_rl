import uvicorn
from openenv_core import create_web_interface_app
from server.env import FlowStateEnv
from models import BlockAction, BlockObservation

# Wrap it with the official OpenEnv visual web interface
app = create_web_interface_app(FlowStateEnv, BlockAction, BlockObservation)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
