import torch
import time
import argparse
from model import Noob, NoobConfig, encode, decode, batch_size, block_size

# 训练相关超参数
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 准备训练数据
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    config = NoobConfig()
    model = Noob(config)
    m = model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_time = time.time()

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(m)
            elapsed_time = time.time() - start_time
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                  f"elapsed time {elapsed_time:.2f}s, remaining batches {max_iters - iter}")

        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final evaluation
    losses = estimate_loss(m)
    elapsed_time = time.time() - start_time
    print(f"Final step {max_iters-1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
          f"elapsed time {elapsed_time:.2f}s")

    # Generate some text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\nGenerated text:")
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

    return m

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train language model')
    parser.add_argument('--save_dir', type=str, default='noob_model',
                        help='Directory to save the model')
    args = parser.parse_args()

    model = train()
    model.save_pretrained(args.save_dir)
    print(f"Model saved to {args.save_dir}")