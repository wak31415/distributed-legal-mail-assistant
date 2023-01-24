import streamlit as st

import syft as sy
from syft.core.adp.data_subject import DataSubject
from syft.core.tensor.smpc.mpc_tensor import MPCTensor
import transformers
from captum.attr import LayerIntegratedGradients
import plotly.express as px 
import tokenizers
import torch

from argparse import ArgumentParser

@st.cache
def get_args():
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

def construct_input_and_baseline(tokenizer, text):
    max_length = 510
    baseline_token_id = tokenizer.pad_token_id 
    sep_token_id = tokenizer.sep_token_id 
    cls_token_id = tokenizer.cls_token_id 

    text_ids = tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False)
   
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    token_list = tokenizer.convert_ids_to_tokens(input_ids)

    baseline_input_ids = [cls_token_id] + [baseline_token_id] * len(text_ids) + [sep_token_id]
    return torch.tensor([input_ids], device='cpu'), torch.tensor([baseline_input_ids], device='cpu'), token_list

def summarize_attributions(attributions):

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    
    return attributions

sy.logger.remove()

args = get_args()
model = get_model()
predicted_labels = {"labels":[], "scores":[]}


def model_output(inputs):
    return model(inputs)[0]

st.title("Private Smart Email Assistant")
st.write(f"Connected to server on {args.server_ip}:{args.server_port}")

with st.sidebar:
    text = st.text_area("List of classification labels, one per line")
    labels = get_labels(text)

email = st.text_area("Email to analyze")

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        predicted_labels = model(email, labels, multi_label=True, )

st.multiselect("Classification",labels, [predicted_labels["labels"][i] for i, score in enumerate(predicted_labels["scores"]) if score > 0.4])

with st.expander("View more detailed statistics"):
    st.bar_chart(predicted_labels, x="labels", y="scores")