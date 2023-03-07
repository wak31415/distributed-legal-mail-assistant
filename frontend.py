import streamlit as st

import syft as sy
from syft.core.adp.data_subject import DataSubject
from syft.core.tensor.smpc.mpc_tensor import MPCTensor
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients
import tokenizers
import torch

import pickle
import socket

from argparse import ArgumentParser
import settings
from mpc.networking import send, receive
import time

@st.cache
def get_args():
    """Get argument parser.
    """
    parser = ArgumentParser()
    parser.add_argument('--server-ip', type=str)
    parser.add_argument('--server-port', type=int)
    return parser.parse_args()

@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None}, allow_output_mutation=True)
def get_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-small', output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-small')
    return model, tokenizer

@st.cache
def get_labels(text):
    labels = text.split("\n")
    labels = [x.strip() for x in labels]
    return labels

sy.logger.remove()

args = get_args()
model, tokenizer = get_model_and_tokenizer()

st.title("Private Smart Email Assistant")
st.write(f"Connected to server on {args.server_ip}:{args.server_port}")

with st.sidebar:
    text = st.text_area("List of classification labels, one per line",
                        placeholder="e.g.:\nerasure\ncancellation\nrectification\nurgent\n...",
                        height=200)
    labels = get_labels(text)

email = st.text_area("Email to analyze")
predicted_labels = {"labels": [], "scores": []}

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        # extract output of last hidden layer of pretrained model
        features = tokenizer([email] * len(labels), [f'This example is {label}.' for label in labels],  padding=True, truncation=True, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            output = model(**features)
        l = model.dropout(model.pooler(output.hidden_states[-1]))

        # first, let the MPC backend know how many different labels there are
        send(len(labels), settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_PORT)
        send(len(labels), settings.HOST_MPC_BACKEND_HOST, settings.HOST_MPC_BACKEND_PORT)
        time.sleep(0.05)
        # send data to MPC BACKEND
        send(l, settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_PORT)

        # receive prediction from MPC backend
        scores = receive(settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_RETURN_PORT)
        
        scores = scores[..., 1]
        scores = scores.tolist() # turn into a python list      
        predicted_labels = {"labels": labels, "scores": scores}

st.multiselect("Classification",labels, [predicted_labels["labels"][i] for i, score in enumerate(predicted_labels["scores"]) if score > 0.4])

with st.expander("View more detailed statistics"):
    st.bar_chart(predicted_labels, x="labels", y="scores")

# if correction, send to server for weight updating with label correction tag