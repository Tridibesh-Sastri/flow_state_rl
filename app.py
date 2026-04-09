import uvicorn
from openenv_core import create_web_interface_app
from env import FlowStateEnv
from models import BlockAction, BlockObservation

# Wrap it with the official OpenEnv visual web interface
app = create_web_interface_app(FlowStateEnv, BlockAction, BlockObservation)

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
