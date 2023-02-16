# code to generate weights.pth
import torch
from transformers import AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-small', output_hidden_states=True)
torch.save(model.classifier, "weights.pth")

# code to generate model.pth
class OutputLayer(nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()
        self.linear = torch.load('weights.pth')
 
    def forward(self, x):
        logits = self.linear(x)
        entailment_id = 1
        contradiction_id = 0
        entail_contr_logits = logits[..., [contradiction_id, entailment_id]]
        scores = F.softmax(entail_contr_logits, dim=1)
        return scores

crypten.init()
crypten.common.serial.register_safe_class(OutputLayer)

SERVER = 0

plaintext_model = OutputLayer()

# create a dummy input with the same shape as the model input
dummy_input = torch.empty((1, 768))

# construct a crypten network with the trained model and dummy_input
private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)

# encrypt the CrypTen network
private_model.encrypt(src=SERVER)
print("Model successfully encrypted:", private_model.encrypted)
crypten.save(private_model, "model.pth")