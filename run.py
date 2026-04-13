import time
_start = time.time()

from dotenv import load_dotenv
load_dotenv()

from src.agent import FeatureAgent

if __name__ == "__main__":
    agent = FeatureAgent(time_budget=575)
    agent.run()
