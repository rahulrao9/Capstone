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

with open('res.json', 'r') as file:
    res = json.load(file)

# Result list to store computed X and Y values
result = []

depth = 0

# Iterate over each item in the dictionary
for key, points in res.items():
    key_parts = key.split(".")[0]
    key_parts = key.split('_')
    x = int(key_parts[-3][1:])  # Remove the 'x' and convert to int
    y = int(key_parts[-2][1:])  # Remove the 'y' and convert to int
    
    # # If the points list is not empty, get the first point as (px, py)
    # if points:
    #     py, px = points[0]
        
    #     # Calculate X and Y using the formula
    #     x = int((x - 225) + (px * 0.375))
    #     y = int((y - 300) + (py * 0.375))
        
        # Store result in the dictionary
    result.append({'x': x, 'y': y, "z":depth})

    print(result)

# Dump result to a JSON file named weeds.json
# with open('weeds.json', 'w') as json_file:
#     json.dump(result, json_file, indent=4)

print("Data has been saved to weeds.json")

if __name__ == "__main__":
    if len(result) != 0:
        log_filename = "./farmbot_run.log"
        logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

        # Load weed coordinates from a JSON file
        with open("./weeds.json", "r") as f:
            WEEDCOORDINATES = json.load(f)

        creds = "./credentials.json"
        
        logging.info("Starting kill mechanism")
        print("weed co",WEEDCOORDINATES)
        for location in WEEDCOORDINATES:
            logging.info(f"Processing location: {location}")
            
            # Move action
            logging.info(f"Starting move action for {location}")
            run_with_timeout(logged_run, [creds, location, "move"], 30)
            logging.info(f"Move action completed for {location}")
            print("hello2")
            remover = location
            remover["z"] -= 480

            logging.info(f"Starting remove action for {location}")
            run_with_timeout(logged_run, [creds, remover, "move"], 30)
            logging.info(f"Remove action completed for {location}")  

            logging.info(f"Complete action for {location}")
            run_with_timeout(logged_run, [creds, location, "move"], 30)
            logging.info(f"Completed action for {location}")

        logging.info("Weeds eliminated")
        print("Weeds eliminated")

    else:
        print("No weeds detected!")

    logging.info(f"Complete action for {location}")
    run_with_timeout(logged_run, [creds, location, "home"], 30)
    logging.info(f"Completed action for {location}")