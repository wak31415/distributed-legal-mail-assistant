import transformers
from argparse import ArgumentParser
from .data import get_data

# get data directory from arguments
parser = ArgumentParser()
parser.add_argument("-d", "--data-dir", type=str, required=True)
args = parser.parse_args()
data_dir = args.data_dir

model = transformers.pipeline("zero-shot-classification",model="cross-encoder/nli-deberta-v3-small", device=0)
data = get_data(data_dir)

sequences = [x["text"] for x in data]
target_labels = [x["labels"] for x in data]
labels = list(set([x for label in target_labels for x in label]))
print(labels)

batched_seq = []
batch_size = 16
for i in range(0,len(sequences)-batch_size+1,batch_size):
    batch = sequences[i:i+batch_size]
    batched_seq.append(batch)

print("running model")

predictions = model(sequences, labels, batch_size=batch_size, multi_label=True)

threshold = 0.4
predictions = [[sample["labels"][i] for i, score in enumerate(sample["scores"]) if score > threshold] for sample in predictions]

pos_acc, neg_acc = 0, 0
for pred, target in zip(predictions, target_labels):
    pos_acc += len(set(pred).intersection(set(target))) / len(target)
    neg_acc += len(set(pred).difference(set(target))) > 0

print(f"pos_acc: {pos_acc / len(target_labels)}, neg_acc: {neg_acc/len(target_labels)}")