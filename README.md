# Distributed Legal Email Assistant

## Set up environment

Clone the repository. Using a python environment manager of your choice, install the requirements from `requirements.txt`, e.g.

```bash
git clone git@github.com:[username]/distributed-legal-mail-assistant.git
cd distributed-legal-mail-assistant

conda create -n email-paralegal python=3.9
conda activate email-paralegal
pip install -r requirements.txt
```

## Run Client

So far, only inference on a pretrained model is supported. To run the UI and perform classifications on emails, run

```bash
streamlit run source/client/run.py -- --server-ip 0.0.0.0 --server-port 0000
```

Note: `--server-ip` and `--server-port` have no purpose yet.

## Run Inference on Test Data

To run the model on all exisiting test data, simply run

```bash
python -m utils.testing -d data
```

## Data Structure

Each email is stored as a separate file in `data`, and admits the following format:

```txt
[comma separated list of true labels]
multi
line
email
```