import json
from farmbot import Farmbot, FarmbotToken

class MyMoveHandler:
    def on_connect(self, bot, mqtt_client):
        for plant in plant_cord:
            x, y, z = plant["x"], plant["y"], plant["z"]
            request_id = bot.move_absolute(x, y, z)
            print(f"Moving to plant at ({x}, {y}, {z}). Request ID: {request_id}")


    def on_change(self, bot, state):
        pass

    def on_log(self, bot, log):
        print(f"Log: {log['message']}")

    def on_response(self, bot, response):
        print(f"Successful request: {response.id}")

    def on_error(self, bot, response):
        print(f"Failed request: {response.id}, Errors: {response.errors}")

class MyPhotoHandler:
    def on_connect(self, bot, mqtt_client):
        # bot.toggle_pin(7)
        request_id = bot.take_photo()
        print(f"Taking photo. Request ID: {request_id}")

    def on_change(self, bot, state):
        pass

    def on_log(self, bot, log):
        print(f"Log: {log['message']}")

    def on_response(self, bot, response):
        print(f"Successful request: {response.id}")
        # bot.toggle_pin(7)

    def on_error(self, bot, response):
        print(f"Failed request: {response.id}, Errors: {response.errors}")
        # bot.toggle_pin(7)

class MyHomeHandler:
    def on_connect(self, bot, mqtt_client):
        # bot.toggle_pin(7)
        request_id = bot.find_home()
        print(f"Moving robot to home position. Request ID: {request_id}")

    def on_change(self, bot, state):
        pass

    def on_log(self, bot, log):
        print(f"Log: {log['message']}")

    def on_response(self, bot, response):
        print(f"Successful request: {response.id}")

    def on_error(self, bot, response):
        print(f"Failed request: {response.id}, Errors: {response.errors}")

def run(credentials, location, type):
    global plant_cord
    plant_cord = [location,]
    # Load credentials from a JSON file
    with open(credentials, "r") as f:
        creds = json.load(f)
        email = creds["email"]
        password = creds["password"]
        server = creds["server"]

    # Download token and create FarmBot instance
    raw_token = FarmbotToken.download_token(email, password, server)
    fb = Farmbot(raw_token)

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