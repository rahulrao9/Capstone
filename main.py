from phototaker import MyHomeHandler, MyMoveHandler, MyPhotoHandler
import multiprocessing
import json
import logging
from datetime import datetime
from contextlib import redirect_stdout
from farmbot import Farmbot, FarmbotToken

# Configure logging
log_filename = "./farmbot_run.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load plant coordinates from a JSON file
with open("./locations.json", "r") as f:
    PLANTCOORDINATES = json.load(f)

# Load credentials from a JSON file
creds = "./credentials.json"
with open(creds, "r") as f:
    creds = json.load(f)
    email = creds["email"]
    password = creds["password"]
    server = creds["server"]


def run(fb,type):

    if type == "move":
        # Connect to FarmBot with the custom handler
        handler = MyMoveHandler()
        fb.connect(handler)
    elif type == "photo":
        handler = MyPhotoHandler()
        fb.connect(handler)
    elif type == "home":
        handler = MyHomeHandler()
        fb.connect(handler)

def logged_run(fb, action):
    with open(log_filename, 'a') as log_file:
        with redirect_stdout(log_file):
            run(fb, action)

def run_with_timeout(target, args, timeout):
    p = multiprocessing.Process(target=target, args=args)
    p.start()
    p.join(timeout)
    if p.is_alive():
        logging.warning(f"Process timed out for {args[1]} with action {args[2]}")
        p.terminate()
        p.join()

if __name__ == "__main__":
    
    logging.info("Starting plant photography process")

    # Download token and create FarmBot instance
    raw_token = FarmbotToken.download_token(email, password, server)
    fb = Farmbot(raw_token)

    for location in PLANTCOORDINATES:

        global plant_cord
        plant_cord = [location,]
        logging.info(f"Processing location: {location}")

        # Move action
        logging.info(f"Starting move action for {location}")
        run_with_timeout(logged_run, [fb, "move"], 30)
        logging.info(f"Move action completed for {location}")

        # Photo action
        logging.info(f"Starting photo action for {location}")
        run_with_timeout(logged_run, [fb, "photo"], 30)
        logging.info(f"Photo action completed for {location}")

        logging.info(f"Moving to home")
        run_with_timeout(logged_run, [fb, "home"], 30)
        logging.info(f"Moved to home")       

    logging.info("Plant photography process completed")