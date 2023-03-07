import crypten
import crypten.communicator as comm
import torch
import torch.nn.functional as F
import os
import crypten.communicator as comm
from argparse import ArgumentParser
import logging
import socket
import pickle
from mpc.networking import *
from mpc.model import OutputLayer
import settings

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
        "RENDEZVOUS": f"tcp://{args.master_ip}:{args.master_port}",
        "MASTER_ADDR": args.master_ip,
        "MASTER_PORT": str(args.master_port),
        "BACKEND": args.backend,
        "GLOO_SOCKET_IFNAME": args.socket,
    }
    for key, val in communicator_args.items():
        os.environ[key] = str(val)

    crypten.init()
    crypten.common.serial.register_safe_class(OutputLayer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--master-ip", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=9003)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--socket", type=str, default=None)
    args = parser.parse_args()
    print("args")
    init_environment(args)

    print("hey")

    # load a crypten model and encrypt it
    private_model = crypten.load("model.pth")
    private_model.encrypt(src=SERVER)
    print("Model successfully encrypted:", private_model.encrypted)

    # training params
    loss = crypten.nn.BCEWithLogitsLoss()
    learning_rate = 1e-3
    num_epochs = 10

    while True:
        # TODO: if each iteration is a new msg from the front end, we will need some
        # sort of tag for each msg so that we can know if it's an inference 
        # request or a label correction. Note in latter case, front end will send
        # two tensors: the input, and the target
        if comm.get().get_rank() == SERVER:
            num_categories = receive(settings.HOST_MPC_BACKEND_HOST, settings.HOST_MPC_BACKEND_PORT)
            if num_categories is None:
                continue
            logging.info(f"Received data from frontend: '{num_categories}'")
        else:
            num_categories = receive(settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_PORT)
            if num_categories is None:
                continue
            logging.info(f"Received data from frontend: '{num_categories}'")

        data = torch.zeros((num_categories, 768))

        if comm.get().get_rank() == CLIENT:
            data = receive(settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_PORT)
            if data is None:
                continue
            logging.info(f"Received tensor from frontend: {data}")

        
        # if inference request...
        # inference with private model
        print(data.shape)
        x_enc = crypten.cryptensor(data, src=CLIENT)
        private_model.eval()
        pred_enc = private_model(x_enc)

        # client-side send decrypted labels back to frontend
        scores = pred_enc.get_plain_text()
        # use socket to send to frontend. IMPORTANT: include receiving code in `client/run.py`
        if comm.get().get_rank() == CLIENT:
            send(scores, settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_RETURN_PORT)

        # if label correction
        # below is training code
        # x_enc = ???
        # target_enc = ???
        """
        private_model.train() # Change to training mode
        for _ in range(num_epochs):
            pred_enc = private_model(x_enc)
            loss_value = loss(output, target_enc)
            
            # set gradients to zero
            private_model.zero_grad()

            # perform backward pass
            loss_value.backward()

            # update parameters
            private_model.update_parameters(learning_rate) 
            
            # examine the loss after each epoch
            # print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss_value.get_plain_text()))
        crypten.save(private_model, "model.pth") # save model
        """