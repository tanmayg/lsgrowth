import logging
import os
from datetime import datetime

# Create a log directory name with just the date
log_dir_name = datetime.now().strftime('%m_%d_%Y')
logs_path = os.path.join(os.getcwd(), "logs", log_dir_name)

# Ensure the directory exists
os.makedirs(logs_path, exist_ok=True)

# Now create the log file name with the time included
LOG_FILE = f"{datetime.now().strftime('%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Setup the basic configuration for logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Adding StreamHandler to log to console as well
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

# Test logging
logging.info("This is an info message.")
