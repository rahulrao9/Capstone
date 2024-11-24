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