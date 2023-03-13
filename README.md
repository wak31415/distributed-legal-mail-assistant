# Distributed Legal Email Assistant

## How to Run (locally)

### Set up environment

Clone the repository. Using a python environment manager of your choice, install the requirements from `requirements.txt`, e.g.

```
git clone git@github.com:wak31415/distributed-legal-mail-assistant.git
cd distributed-legal-mail-assistant

conda create -n email-paralegal python=3.9
conda activate email-paralegal
pip install -r requirements.txt
```

### Run MPC Backend for Client

This is the process that handles all of the MPC communication with the server (SIFTIT), and runs on the infrastructure of the law firm.

```bash
python run_mpc.py --master-ip '127.0.0.1' --rank 1 --socket lo
```

### Run MPC Backend for Server

This is the process that handles all of the MPC communication with the client (law firm), and runs on our servers (SIFTIT). Note that the two scripts are the same for client and server because of how CrypTen works - this does not mean that they have access to the same data.

```bash
python run_mpc.py --master-ip '127.0.0.1' --rank 0 --socket lo
```

### Run Frontend

We use Streamlit for the frontend. To launch the process, simply run:

```
streamlit run frontend.py
```

### Run Inference on Test Data

To run the model on all exisiting test data, simply run

```
python -m utils.testing -d data

```

## How to Run (distributed)

While many of the foundations of this are set up, we have not been able to test it due to very limited documentation on this in CrypTen. We emphasize that CrypTen is **not designed for production use**, and is **not actively maintained**. 

If you want to deploy this to a distributed setup, you can try changing `[settings.py](http://settings.py)` as well as some of the flags. Some fixing in the code may be necessary to make it work.

## Data Structure

Each email is stored as a separate file in `data`, and admits the following format:

```
[comma separated list of true labels]
multi
line
email
```

## Appendix

### Pre-trained Model

For the pre-trained natural language processing model, we use the cross-encoder/nli-deberta-v3-small [[https://huggingface.co/cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small)] from the Hugging Face library.

This model was pre-trained on the SNLI and MultiNLI datasets using the Hugging Face SentenceTransformers Cross-Encoder class. It takes two sentences as input and outputs three scores representing the labels: contradiction, entailment, neutral.

The model is 568 MB.

The model’s first layer is an embedding layer, which generates word embeddings with [768] features. The model then has 6 encoder DebertaV2Layer[s] which each take an input with [768] features, perform an attention function on the input, and a GELU activation function before outputting [768] features. The model then pools the output of the final encoder layer, taking an input with [768] features and outputting [768] features. Finally, the model has a linear layer which classifies the final embeddings of size [768], outputting [3] features to represent the scores for each label.

This model can be used for zero-shot-classification which was suitable for our task because you can specify new labels at point of inference without retraining the model. The probability that a given sentence $s$ has a given label $l$ **is the probability of entailment between $s$ and the sentence “This example is $l$.” (when we discount neutrality as a possibility).

### Our adaptation of the model

To adapt the model for MPC, we simply replace the final linear layer (the classifier) with an MPC linear layer. This minimises the extra potential computation cost that comes with MPC. The computations before the MPC layer are done on the client’s machine, and the MPC layer computations are carried out between the client’s machine and the server.

We will perform transfer learning on this final MPC layer only - the weights of the pre-trained model in all other layers remain unchanged when we train. We will perform online training after deployment of our model to the client. The model will learn from the client’s alterations to the labels the model gives them.

### MPC

Secure Multi-Party Computation is a method which attempts to solve the following problem: Given $n$ parties ($P_1, …, P_n$), each holding private information $d_1, …, d_n \in A$ respectively, intend to compute a publicly known function $f:A^n \rightarrow B$ without requiring any of the parties to reveal their private information to each other or anyone else. 

To implement the MPC component, we use the [CrypTen](https://arxiv.org/pdf/2109.00984.pdf) library. We refer to section 5 of the paper for the algorithms for the secure computations. CrypTen adopts a semi-honest model, meaning it is secure against honest-but-curious adversaries - adversaries which follow the protocol honestly, but are interested in learning the private information of the other players. Note that adversaries in this context do not refer to 3rd party attackers, but rather mean the involved parties (law firms and us).

We restate the following result from the paper:

> **Lemma 1.** The CrypTen secure-computation protocol is secure against information leakage against any static passive adversary corrupting up to |P| − 1 of the |P| parties involved in the computation.
> 

Note that the chosen protocol, and therefore the application, is **not** secure against malicious adversaries (i.e., law firms) - that is, a law firm could decide not to follow protocol. Such actions would go undetected, and the consequences thereof are uncertain. While there are actively-secure protocols (secure against malicious adversaries), there is a performance trade-off.

### Our infrastructure

Our code consists of three core components: the client’s front-end, the client-side MPC process, and the server-side MPC process.

For inference, the front-end receives the labels and input text from the user and runs the pre-trained model on them, up until the final linear layer. It then passes this output to the client-side MPC process, which encrypts the output. At the same time, the server-side MPC process encrypts the final MPC layer, and the two MPC processes perform MPC to obtain the output. This output is then sent back to the front-end, which does some postprocessing to display the each of the label likelihoods for the input text.

For training, something analogous runs, i.e. once the labels are confirmed, this information is sent to the client-side MPC process and encrypted, so MPC training can be performed with the encrypted model on the server-side MPC process.