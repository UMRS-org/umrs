import socket

hostname = socket.gethostbyaddr("192.168.42.134")[0]
print(hostname)
