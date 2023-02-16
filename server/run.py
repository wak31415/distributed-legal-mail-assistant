import crypten
import crypten.communicator as comm
import torch
import torch.nn.functional as F
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

class OutputLayer(nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()
        self.linear = torch.load('weights.pth')
 
    def forward(self, x):
        logits = self.linear(x)
        entailment_id = 1
        contradiction_id = -1
        entail_contr_logits = logits[..., [contradiction_id, entailment_id]]
        scores = F.softmax(entail_contr_logits, dim=1)
        return scores

    def save(self):
        torch.save(self.linear, "weights.pth")

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

    model = OutputLayer()

    # create a dummy input with the same shape as the model input
    dummy_input = torch.empty((1, 768))

    # construct a CrypTen network with the trained model and dummy_input
    private_model = crypten.nn.from_pytorch(model, dummy_input)

    # encrypt it
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
        l = torch.tensor([0])
        if comm.get().get_rank() == CLIENT:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
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
        
        # if inference request...
        # inference with private model
        x_enc = crypten.cryptensor(l, src=CLIENT)
        private_model.eval()
        pred_enc = private_model(x_enc)

        # TODO: client-side send decrypted labels back to frontend
        if comm.get().get_rank() == CLIENT:
            labels = pred_enc.get_plain_text()
            # use socket to send to frontend. IMPORTANT: include receiving code in `client/run.py`
            pass

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
            # print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss_value.get_plain_text()))"""