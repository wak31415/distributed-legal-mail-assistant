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
from copy import copy

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
    init_environment(args)

    # load a crypten model and encrypt it
    private_model = crypten.load("model.pth")
    private_model.train()

    # training params
    loss = crypten.nn.BCEWithLogitsLoss()
    learning_rate = 1e-1 # high for illustrative testing purposes
    num_epochs = 10

    while True:
        private_model.encrypt(src=SERVER)
        print("Model successfully encrypted:", private_model.encrypted)

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

        # server needs a tensor in the shape of the data received by the client
        # to perform the MPC
        data = torch.zeros((num_categories, 768))

        if comm.get().get_rank() == CLIENT:
            data = receive(settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_PORT)
            if data is None:
                continue
            logging.info(f"Received tensor from frontend: {data}")

        # inference with private input and model
        print(data.shape)
        x_enc = crypten.cryptensor(data, src=CLIENT)
        pred_enc = private_model(x_enc)

        # client-side send decrypted labels back to frontend
        scores = copy(pred_enc).get_plain_text()
        if comm.get().get_rank() == CLIENT:
            send(scores, settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_RETURN_PORT)

        # get confirmed/updated labels from frontend
        y_true = torch.zeros((num_categories, 2))
        if comm.get().get_rank() == CLIENT:
            y_true = receive(settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_PORT)
            print(y_true)
        
        y_true_enc = crypten.cryptensor(y_true, src=CLIENT)
        print(copy(pred_enc).get_plain_text())
        
        # train with confirmed/updated labels
        for _ in range(num_epochs):    
            # perform backward pass
            loss_ = loss(pred_enc, y_true_enc)
            private_model.zero_grad() # set gradients to zero
            loss_.backward()
            
            # update parameters
            private_model.update_parameters(learning_rate)
            
            logging.info("Loss: {0:.4f}".format(loss_.get_plain_text()))
        
        # save updated parameters
        private_model.decrypt()
        logging.info("Successfully decrypted model")
        if comm.get().get_rank() == SERVER:
            torch.save(private_model, "model.pth")
            logging.info("Saved updated model")