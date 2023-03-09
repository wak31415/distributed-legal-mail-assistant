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

if 'labels_changed' not in st.session_state:
    st.session_state.labels_changed = False

if 'done_analyzing' not in st.session_state:
    st.session_state.done_analyzing = False

if 'classified_label_list' not in st.session_state:
    st.session_state.classified_label_list = []

if 'label_selector' not in st.session_state:
    st.session_state.label_selector = []

if 'predicted_labels' not in st.session_state:
    st.session_state.predicted_labels = {}

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

def invoke_training_step():
    print("Predicted:",st.session_state.classified_label_list)
    print("Updated:",st.session_state.label_selector)
    st.session_state.labels_changed = True

sy.logger.remove()

args = get_args()
model, tokenizer = get_model_and_tokenizer()

st.title("Private Smart Email Assistant")
st.write(f"Connected to server on {args.server_ip}:{args.server_port}")

with st.sidebar:
    text = st.text_area("List of classification labels, one per line",
                        placeholder="e.g.:\nerasure\nclaim cancellation\nrectification\nurgent\n...",
                        height=200)
    labels = get_labels(text)

email = st.text_area("Email to analyze")
label_selector = st.empty()

if st.button("Analyze"):
    label_selector.empty()
    st.session_state.done_analyzing = False
    st.session_state.classified_label_list = []
    st.session_state.predicted_labels = {}
    # st.session_state.label_selector = st.empty()
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

        st.session_state.predicted_labels = predicted_labels
        st.session_state.classified_label_list = [predicted_labels["labels"][i] for i, score in enumerate(predicted_labels["scores"]) if score > 0.4]

    st.session_state.done_analyzing = True

if st.session_state.done_analyzing:
    with st.form(key='results'):
        st.text(st.session_state.classified_label_list)
        # label_selector.empty()
        out = label_selector.multiselect("Classification", labels, 
                                    st.session_state.classified_label_list)
        st.session_state.label_selector = out

        with st.expander("View more detailed statistics"):
            st.bar_chart(st.session_state.predicted_labels, x="labels", y="scores")

        if st.form_submit_button(label="Confirm or Update Labels"):
            indices = list(map(lambda x: labels.index(x), st.session_state.label_selector))
            
            updated_output_tensor = torch.tensor([1.0, 0.0]).repeat(len(labels), 1)
            updated_output_tensor[indices] = torch.tensor([0.0, 1.0])

            send(updated_output_tensor, settings.CLIENT_MPC_BACKEND_HOST, settings.CLIENT_MPC_BACKEND_PORT)

            st.text(updated_output_tensor)
            st.text(f"Predicted: {st.session_state.classified_label_list}")
            st.text(f"Updated: {st.session_state.label_selector}")
            st.session_state.done_analyzing = False

# if correction, send to server for weight updating with label correction tag
# if st.session_state.labels_changed:
#     st.info("Updating model weights")
#     st.session_state.labels_changed = False