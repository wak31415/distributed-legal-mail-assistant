import streamlit as st

import syft as sy
from syft.core.adp.data_subject import DataSubject
from syft.core.tensor.smpc.mpc_tensor import MPCTensor
import transformers
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
def get_model():
    return transformers.pipeline("zero-shot-classification",model="cross-encoder/nli-deberta-v3-small", device=0)

@st.cache
def get_labels(text):
    labels = text.split("\n")
    labels = [x.strip() for x in labels]
    return labels

sy.logger.remove()

args = get_args()
model = get_model()
predicted_labels = {"labels":[], "scores":[]}

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
        # TODO: modify this to get the final layer embeddings of the email
        predicted_labels = model(email, labels, multi_label=True, )

        # pickle the data for sending
        pickled_l = pickle.dumps(l)

        # this section handels network communication with the client-side MPC script (server/run.py)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(pickled_l)

st.multiselect("Classification",labels, [predicted_labels["labels"][i] for i, score in enumerate(predicted_labels["scores"]) if score > 0.4])

with st.expander("View more detailed statistics"):
    st.bar_chart(predicted_labels, x="labels", y="scores")