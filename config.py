import os

# Automatically set the base directory to the location of this config file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define key directories relative to BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# Ensure required directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, SCRIPTS_DIR]:
    os.makedirs(path, exist_ok=True)
