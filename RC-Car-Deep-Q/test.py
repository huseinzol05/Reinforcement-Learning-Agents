from socketIO_client import SocketIO, BaseNamespace
import numpy as np
import json

def on_car_response(*args):
	print args

socketIO = SocketIO('https://huseinzol05.dynamic-dns.net', 9001)
car_namespace = socketIO.define(BaseNamespace, '/carsystem')
car_namespace.on('carsensor', on_car_response)
while 1:
	car_namespace.emit('carsensor', json.dumps(np.random.randint(1500, size = 4).tolist()))