from rosbridge_library.util import json
from ws4py.client.threadedclient import WebSocketClient
import threading

import json
import time

class RobotRPC(WebSocketClient):

    def __init__(self, url='ws://localhost:9090', ros_node_handler=None):
        self.nh = ros_node_handler
        try:
            super(RobotRPC, self).__init__(url)
            self.connect()
        except:
            self.close()

    def received_message(self, message):
        print(message.data)

    def __del__(self):
        self.close()
  
    def get_pose(self, target, source, time):
        msg={"op": "subscribe",
            "id": "gym_ros",
            "topic": "/tf",
            "queue_length":20}
        self.send(json.dumps(msg))

    def start(self):
        self.run_forever()



if __name__ == '__main__':
    try:
        ws = RobotRPC('ws://10.2.2.18:9090')
        time.sleep(2)
        ws.get_pose(0,0,0)
        ws.start()
    except KeyboardInterrupt:
        ws.close()