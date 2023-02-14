import crypten
import torch
import os
import crypten.communicator as comm
from argparse import ArgumentParser
from distributed_launcher import DistributedLauncher

SERVER = 0
CLIENT = 1

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
    # launcher = DistributedLauncher(args.world_size, args.rank, args.master_ip, args.master_port, args.backend, args.socket, test)
    # launcher.start()
    x_enc = crypten.cryptensor([1,2,3], src=CLIENT)
    y_enc = crypten.cryptensor([4,2,1], src=SERVER)

    z_enc = crypten.where(x_enc < y_enc, 1, 0)

    print(z_enc.get_plain_text())