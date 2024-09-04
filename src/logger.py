# import
import os
import logging
from datetime import datetime

# Log set up
LOG_FILE= f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path= os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(log_path), exist_ok=True)


# Basic Config
logging.basicConfig(
    filename=log_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Test
def main():
     logging.info("Logging has started")

if __name__=="__main__":
   main()