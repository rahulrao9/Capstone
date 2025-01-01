from phototaker import run
import multiprocessing
import json
import logging
from datetime import datetime
from contextlib import redirect_stdout

# Configure logging
log_filename = "./farmbot_run.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load plant coordinates from a JSON file
with open("./locations.json", "r") as f:
    PLANTCOORDINATES = json.load(f)
    print(PLANTCOORDINATES)

def logged_run(creds, location, action):
    with open(log_filename, 'a') as log_file:
        with redirect_stdout(log_file):
            run(creds, location, action)

def run_with_timeout(target, args, timeout):
    p = multiprocessing.Process(target=target, args=args)
    p.start()
    p.join(timeout)
    if p.is_alive():
        logging.warning(f"Process timed out for {args[1]} with action {args[2]}")
        p.terminate()
        p.join()

if __name__ == "__main__":
    creds = "./credentials.json"
    
    logging.info("Starting plant photography process")

    for location in PLANTCOORDINATES:
        print(location)
        logging.info(f"Processing location: {location}")

        # Move action
        logging.info(f"Starting move action for {location}")
        run_with_timeout(logged_run, [creds, location, "move"], 15)
        logging.info(f"Move action completed for {location}")

        # Photo action
        logging.info(f"Starting photo action for {location}")
        run_with_timeout(logged_run, [creds, location, "photo"], 30)
        logging.info(f"Photo action completed for {location}")

        logging.info(f"Moving to home")
        run_with_timeout(logged_run, [creds, location, "home"], 30)
        logging.info(f"Moved to home")       

    logging.info("Plant photography process completed")