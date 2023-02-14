import socket
import pickle

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 9004  # The port used by the server

while True:
    l = input("Enter 3 integers separated by spaces: ")
    l = list(map(lambda x: int(x), l.split()))
    pickled_l = pickle.dumps(l)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(pickled_l)