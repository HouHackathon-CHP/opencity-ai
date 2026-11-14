import time
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(20)
print(f"Testing PhoBERT on CPU with {torch.get_num_threads()} threads...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
m = AutoModel.from_pretrained("vinai/phobert-base-v2")
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
print(f"Model loaded in {time.time()-t0:.2f}s")

texts = ["Đèn đường chiếu sáng bị chập điện nguy hiểm tại ngã tư", "Mặt đường xuất hiện ổ gà sâu gây tai nạn giao thông"] * 16 # batch 32
enc = tok(texts, padding="max_length", max_length=80, return_tensors="pt")

t1 = time.time()
for step in range(5):
    s0 = time.time()
    out = m(**enc)
    loss = out.last_hidden_state.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"Step {step+1}: {time.time()-s0:.2f}s per batch (batch_size=32)")

total = time.time() - t1
print(f"Average time per batch: {total/5:.2f}s")

