from transformers import T5Tokenizer, T5EncoderModel, BertGenerationEncoder, BertTokenizer
import torch
import re
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

transformer_link = "../Pretrain/prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))
model = T5EncoderModel.from_pretrained(transformer_link)
if device==torch.device("cuda"):
  model.to(torch.float32) # only cast to full-precision if no GPU is available
model = model.to(device)
model = model.eval()
tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, legacy=True )

def read(data_file):
    label = []
    sequence = []
    with open(data_file, 'r') as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            if line.startswith('>Positive'):
                label.append(1)
                li = lines[i+1].replace("\n","")
                if (len(li) >= 1000):
                    li = li[0:1000]
                sequence.append(li)
                i += 1
            elif line.startswith('>Negative'):
                label.append(0)
                li = lines[i + 1].replace("\n", "")
                if (len(li) >= 1000):
                    li = li[0:1000]
                sequence.append(li)
                i += 1
            else:
                i += 1
    return label, sequence
data_file = "AP_Ind4.txt"
label, sequence = read(data_file)

import numpy as np
i=0
for seq in sequence:
    print(i)
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(sequence_examples, add_special_tokens=False, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        emb_0 = embedding_repr.last_hidden_state[0, :len(sequence_examples[0])].mean(dim=0)
        with open("embedding/ProtT5_Test4.csv",'a') as f:
            np.savetxt(f, emb_0.cpu().numpy().reshape(1, -1) , delimiter=',')
            torch.cuda.empty_cache()
    i = i + 1
