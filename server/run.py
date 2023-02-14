import crypten
import crypten.communicator as comm
import torch
import os
import crypten.communicator as comm
from argparse import ArgumentParser
from distributed_launcher import DistributedLauncher
import logging
import socket
import pickle

logging.getLogger().setLevel(logging.INFO)

SERVER = 0
CLIENT = 1

FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 9004

def recvall(sock):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while True:
        packet = sock.recv(1024)
        if not packet:
            break
        data.extend(packet)
    return data

def init_environment(args):
    communicator_args = {
        "WORLD_SIZE": str(args.world_size),
        "RANK": str(args.rank),
        # "RENDEZVOUS": f"tcp://{args.master_ip}:{args.master_port}",
        "MASTER_ADDR": args.master_ip,
        "MASTER_PORT": str(args.master_port),
        "BACKEND": args.backend,
        "GLOO_SOCKET_IFNAME": args.socket,
    }
    for key, val in communicator_args.items():
        os.environ[key] = str(val)

    crypten.init()

def test():
    x_enc = crypten.cryptensor([1,2,3], src=CLIENT)
    y_enc = crypten.cryptensor([4,2,1], src=SERVER)

    z_enc = crypten.where(x_enc < y_enc, 1, 0)

    print(z_enc.get_plain_text())



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--master-ip", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=9003)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--socket", type=str, default=None)
    args = parser.parse_args()
    init_environment(args)

    # TODO: load weights from `weights.pth`

    while True:
        l = [0, 0, 0]
        print('hey there again')
        if comm.get().get_rank() == CLIENT:
            print("hi")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                print("hey there")
                s.bind((FRONTEND_HOST, FRONTEND_PORT))
                s.listen()
                conn, addr = s.accept()
                with conn:
                    data = recvall(conn)
                    
            if data:
                l = pickle.loads(data)
                print(l)
            else:
                continue

        # TODO: replace this section with weight updating
        logging.info("x_enc")
        x_enc = crypten.cryptensor(l, src=CLIENT)
        logging.info("y_enc")
        y_enc = crypten.cryptensor([4,2,1], src=SERVER)

        logging.info("z_enc")
        z_enc = crypten.where(x_enc < y_enc, 1, 0)
        logging.info("done")
        # END TODO

        # TODO: client-side send decrypted labels back to frontend
        if comm.get().get_rank() == CLIENT:
            # logging.info(z_enc.get_plain_text())
            labels = pred_enc.get_plain_text()
            # use socket to send to frontend. IMPORTANT: include receiving code in `client/run.py`
            pass

        # TODO: save updated weights to `weights.pth`