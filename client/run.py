import streamlit as st

import syft as sy
from syft.core.adp.data_subject import DataSubject
from syft.core.tensor.smpc.mpc_tensor import MPCTensor
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients
import tokenizers
import torch

from argparse import ArgumentParser

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

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        # extract output of last hidden layer of pretrained model
        features = tokenizer([email] * len(labels), [f'This example is {label}.' for label in labels],  padding=True, truncation=True, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            output = model(**features)
        l = model.dropout(model.pooler(output.hidden_states[-1]))

        # pickle the data for sending
        pickled_l = pickle.dumps(l)

        # send to client-side MPC script (server/run.py)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(pickled_l)

        # front end should receive output of MPC here - assume this is a tensor called scores

        scores = scores[..., 1]
        scores = scores.tolist() # turn into a python list      
        predicted_labels = {"labels": labels, "scores": scores}

st.multiselect("Classification",labels, [predicted_labels["labels"][i] for i, score in enumerate(predicted_labels["scores"]) if score > 0.4])

with st.expander("View more detailed statistics"):
    st.bar_chart(predicted_labels, x="labels", y="scores")