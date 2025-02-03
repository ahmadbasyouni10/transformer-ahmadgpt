import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 4 # how many sequences every forward backwards pass for transformer
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
str_to_int = {c:i for i, c in enumerate(vocab)}
int_to_str = {i:c for i, c in enumerate(vocab)}
encode = lambda x: [str_to_int[c] for c in x]
decode = lambda x: "".join([int_to_str[n] for n in x])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
  # generate a small batch of data inputs x and targets y
  data = train_data if split == "train" else test_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  # will get a 4 by 8 tensor
  # 4 sequences, each sequence size 8
  # the y at each corresponding will be the target the x needs
  # to predict next
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+1+block_size] for i in ix])
  x, y = x.to(device), y.to(device)
  return x,y

@torch.no_grad() # memory efficient since no backprop will be done 
def estimate_loss():
   out = {}
   model.eval()
   for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
   model.train()
   return out


class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
  
  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
    # targets = targets.view(-1) let pytorch lay it out
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for __ in range(max_new_tokens):
      # doing this will pass the xb into the forward method
      logits, loss = self(idx)

      # We want the logits for the last token only since generating
      # based on current context
      logits = logits[:, -1, :]

      # normalize using softmax 0 to 1 on dim 1 because the logits are in tensor
      # [[......]]
      probs = F.softmax(logits, dim = -1)
 
      # torch.multinomial samples one token per batch element
      # based on probs in probs
      idx_next = torch.multinomial(probs, num_samples=1)
      val = int(idx_next[0][0].item())

      idx = torch.cat((idx, idx_next), dim=1)
    
    # Return final sequence after generating max_new_tokens
    return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}, train loss: {losses['train']:.4f}, test loss: {losses['test']:.4f}")
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
