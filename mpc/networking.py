import socket
import pickle
import logging

def send(data, host, port):
    pickled_data = pickle.dumps(data)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(pickled_data)

def recvall(sock):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while True:
        packet = sock.recv(1024)
        if not packet:
            break
        data.extend(packet)
    return data

def receive(host, port):
    logging.info("Waiting for incoming")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        logging.info("Listening...")
        s.listen()
        conn, addr = s.accept()
        logging.info("Connection accepted. Receiving...")
        with conn:
            data = recvall(conn)
            
    if data:
        data = pickle.loads(data)

    logging.info("Done.")
    return data