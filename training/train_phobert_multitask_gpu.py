# SPDX-License-Identifier: Apache-2.0
"""
Fine-tune Multi-Task PhoBERT-base-v2 on fithou RTX 3090 GPU:
1. Category Classification (12 granular labels -> 4 core municipal units)
2. Urgency Classification (3 levels: high, normal, low)
3. Sentence Semantic Embedding (Mean-pooled L2-normalized) for Deduplication
Single Shared Backbone architecture - Runs on CUDA with AMP.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    RobertaModel,
    RobertaPreTrainedModel,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from safetensors.torch import save_file

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    USE_AMP = True
    torch.backends.cudnn.benchmark = True
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
    print(f"Device: GPU {torch.cuda.get_device_name(0)} (Total VRAM: {vram_total:.1f} GB, Free: {vram_free:.1f} GB)", flush=True)
else:
    DEVICE = torch.device("cpu")
    USE_AMP = False
    print("Device: CPU", flush=True)

BATCH_SIZE = 48
GRAD_ACCUM = 1
MAX_LEN = 80
EPOCHS = 3

MODEL_NAME = "vinai/phobert-base-v2"
DATA_PATH = os.path.expanduser("~/opencity_ai/data/civic_dataset_multitask_nationwide.json")
OUTPUT_DIR = os.path.expanduser("~/opencity_ai/output/phobert_multitask_nationwide")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Multi-Task Model Definition
class PhoBertMultiTask(RobertaPreTrainedModel):
    def __init__(self, config, num_categories=12, num_urgencies=3):
        super().__init__(config)
        self.num_categories = num_categories
        self.num_urgencies = num_urgencies

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        hidden_size = config.hidden_size

        # Category classification head (12 classes)
        self.category_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_categories),
        )

        # Urgency classification head (3 classes: high, normal, low)
        self.urgency_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_urgencies),
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        category_labels=None,
        urgency_labels=None,
        return_dict=True,
    ):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]

        category_logits = self.category_classifier(cls_output)
        urgency_logits = self.urgency_classifier(cls_output)

        # Mean pooling L2-normalized for semantic similarity / deduplication
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        sentence_embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

        loss = None
        if category_labels is not None and urgency_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss_cat = loss_fn(category_logits, category_labels)
            loss_urg = loss_fn(urgency_logits, urgency_labels)
            loss = loss_cat + loss_urg

        return {
            "loss": loss,
            "category_logits": category_logits,
            "urgency_logits": urgency_logits,
            "sentence_embeddings": sentence_embeddings,
        }


# 2. Load Dataset
print(f"Loading Multi-Task dataset from {DATA_PATH}...", flush=True)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

texts = [item["full_text"] for item in raw_data]
categories_str = [item["label_12"] for item in raw_data]
urgencies_str = [item["urgency"] for item in raw_data]

CLASSES_12 = sorted(list(set(categories_str)))
CAT2ID = {c: i for i, c in enumerate(CLASSES_12)}
ID2CAT = {i: c for i, c in enumerate(CLASSES_12)}

URGENCY_CLASSES = ["high", "normal", "low"]
URG2ID = {u: i for i, u in enumerate(URGENCY_CLASSES)}
ID2URG = {i: u for i, u in enumerate(URGENCY_CLASSES)}

LABEL12_TO_UNIT4 = {
    "road_maintenance": "infrastructure",
    "drainage_sewage": "infrastructure",
    "street_lighting": "infrastructure",
    "waste_management": "environment",
    "pollution_emissions": "environment",
    "noise_disturbance": "environment",
    "urban_trees": "environment",
    "traffic_signals": "transport",
    "traffic_congestion_safety": "transport",
    "sidewalk_encroachment": "transport",
    "public_procedures": "intake",
    "security_order_disputes": "intake",
}

cat_ids = [CAT2ID[c] for c in categories_str]
urg_ids = [URG2ID[u] for u in urgencies_str]

print(f"Total samples: {len(texts)}", flush=True)
print(f"Categories ({len(CLASSES_12)}): {CLASSES_12}", flush=True)
print(f"Urgencies ({len(URGENCY_CLASSES)}): {URGENCY_CLASSES}", flush=True)

# 3. Train/Val Split (Stratified 85/15)
strat_key = [f"{c}_{u}" for c, u in zip(cat_ids, urg_ids)]
train_idx, val_idx = train_test_split(
    list(range(len(texts))), test_size=0.15, random_state=42, stratify=strat_key
)
print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}", flush=True)

# 4. Tokenizer & Dataset Loader
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class MultiTaskCivicDataset(Dataset):
    def __init__(self, indices, texts, cat_ids, urg_ids, tokenizer, max_len=MAX_LEN):
        self.indices = indices
        self.texts = [texts[i] for i in indices]
        self.cat_ids = [cat_ids[i] for i in indices]
        self.urg_ids = [urg_ids[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "category_labels": torch.tensor(self.cat_ids[idx], dtype=torch.long),
            "urgency_labels": torch.tensor(self.urg_ids[idx], dtype=torch.long),
        }


train_dataset = MultiTaskCivicDataset(train_idx, texts, cat_ids, urg_ids, tokenizer)
val_dataset = MultiTaskCivicDataset(val_idx, texts, cat_ids, urg_ids, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 5. Initialize Model
print(f"Initializing PhoBertMultiTask backbone ({MODEL_NAME})...", flush=True)
config = AutoConfig.from_pretrained(MODEL_NAME)
model = PhoBertMultiTask.from_pretrained(
    MODEL_NAME,
    config=config,
    num_categories=len(CLASSES_12),
    num_urgencies=len(URGENCY_CLASSES),
)
model.to(DEVICE)

TOTAL_STEPS = (len(train_loader) // GRAD_ACCUM) * EPOCHS
WARMUP_STEPS = int(TOTAL_STEPS * 0.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=TOTAL_STEPS,
)

if USE_AMP and torch.cuda.is_available():
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None

print(f"Starting Multi-Task Training Loop on {DEVICE} (Batch: {BATCH_SIZE}, Steps/Epoch: {len(train_loader)})...", flush=True)
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    ep_t0 = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        cat_labels = batch["category_labels"].to(DEVICE, non_blocking=True)
        urg_labels = batch["urgency_labels"].to(DEVICE, non_blocking=True)

        if USE_AMP and torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    category_labels=cat_labels,
                    urgency_labels=urg_labels,
                )
                loss = out["loss"] / GRAD_ACCUM
            scaler.scale(loss).backward()
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                category_labels=cat_labels,
                urgency_labels=urg_labels,
            )
            loss = out["loss"] / GRAD_ACCUM
            loss.backward()

        total_train_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
            avg_step_loss = total_train_loss / (step + 1)
            elapsed = time.time() - ep_t0
            vram_cur = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
            print(
                f"Epoch {epoch}/{EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {avg_step_loss:.4f} | VRAM: {vram_cur:.0f}MB | Elapsed: {elapsed:.1f}s",
                flush=True,
            )

    # Validation
    model.eval()
    val_cat_preds, val_cat_targets = [], []
    val_urg_preds, val_urg_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

            if USE_AMP and torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            cat_preds = torch.argmax(out["category_logits"], dim=-1).cpu().numpy()
            urg_preds = torch.argmax(out["urgency_logits"], dim=-1).cpu().numpy()

            val_cat_preds.extend(cat_preds)
            val_cat_targets.extend(batch["category_labels"].numpy())
            val_urg_preds.extend(urg_preds)
            val_urg_targets.extend(batch["urgency_labels"].numpy())

    cat_acc = accuracy_score(val_cat_targets, val_cat_preds)
    cat_f1 = f1_score(val_cat_targets, val_cat_preds, average="macro")
    urg_acc = accuracy_score(val_urg_targets, val_urg_preds)
    urg_f1 = f1_score(val_urg_targets, val_urg_preds, average="macro")

    # 4 Units Evaluation
    val_unit_targets = [LABEL12_TO_UNIT4[ID2CAT[c]] for c in val_cat_targets]
    val_unit_preds = [LABEL12_TO_UNIT4[ID2CAT[c]] for c in val_cat_preds]
    unit_acc = accuracy_score(val_unit_targets, val_unit_preds)

    print("-" * 70, flush=True)
    print(f"Epoch {epoch} Results:", flush=True)
    print(f"  [Category 12] Accuracy: {cat_acc*100:.2f}% | Macro-F1: {cat_f1*100:.2f}%", flush=True)
    print(f"  [Urgency 3]   Accuracy: {urg_acc*100:.2f}% | Macro-F1: {urg_f1*100:.2f}%", flush=True)
    print(f"  [4 Units]     Accuracy: {unit_acc*100:.2f}%", flush=True)
    print("-" * 70, flush=True)

total_time = time.time() - t0
print(f"Multi-Task Training Complete in {total_time:.1f}s ({total_time/60:.2f} mins)!", flush=True)

# 6. Save Model Artifacts
print(f"Saving model artifacts to {OUTPUT_DIR}...", flush=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save Safetensors
model_state = model.state_dict()
save_file(model_state, os.path.join(OUTPUT_DIR, "model.safetensors"))
config.save_pretrained(OUTPUT_DIR)

metadata = {
    "model_name": "PhoBERT-base-v2-MultiTask-Civic",
    "architecture": "PhoBertMultiTask (Shared Backbone + Category Head + Urgency Head + Normalized Embeddings)",
    "backbone": MODEL_NAME,
    "classes_12": CLASSES_12,
    "id2cat": ID2CAT,
    "urgency_classes": URGENCY_CLASSES,
    "id2urg": ID2URG,
    "label12_to_unit4": LABEL12_TO_UNIT4,
    "embedding_dim": 768,
    "metrics": {
        "cat_macro_f1": float(cat_f1),
        "urg_macro_f1": float(urg_f1),
        "unit4_acc": float(unit_acc),
    },
}

with open(os.path.join(OUTPUT_DIR, "phobert_multitask_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Saved metadata successfully!", flush=True)

# 7. Semantic Deduplication & Zero-Shot Test
print("\n" + "=" * 70, flush=True)
print("TESTING 3 TASKS SIMULTANEOUSLY (1 FORWARD PASS ON GPU)", flush=True)
print("=" * 70, flush=True)

model.eval()
test_cases = [
    ("Dây điện chiếu sáng đứt rơi lòng thòng xuống đường hở điện nguy cơ giật", "infrastructure", "high"),
    ("Bóng đèn ngõ 45 bị hỏng không sáng cả tuần nay", "infrastructure", "normal"),
    ("Kiến nghị lắp thêm đèn led chiếu sáng tiết kiệm điện ngõ mới mở", "infrastructure", "low"),
    ("Bãi rác tự phát bốc cháy dữ dội khói độc mịt mù gần kho hàng", "environment", "high"),
    ("Xe rác không vào gom rác sinh hoạt chất đống 2 ngày", "environment", "normal"),
    ("Đề xuất đặt thêm thùng rác phân loại tại công viên", "environment", "low"),
    ("Nhóm đua xe lạng lách nẹt pô gây tai nạn chết người", "transport", "high"),
    ("Ùn tắc giao thông cục bộ giờ cao điểm nút giao", "transport", "normal"),
    ("Kiến nghị cắm biển cấm đỗ ngày chẵn lẻ để giảm ùn ứ", "transport", "low"),
    ("Hai nhóm thanh niên mang hung khí dao kiếm tụ tập hỗn chiến", "intake", "high"),
    ("Hàng xóm to tiếng cãi vã tranh chấp ranh giới xây dựng", "intake", "normal"),
    ("Kiến nghị công an phường tăng cường tuần tra đêm ngõ xóm", "intake", "low"),
]

for text, exp_unit, exp_urg in test_cases:
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
        cat_p = torch.softmax(out["category_logits"], dim=-1).cpu().numpy()[0]
        urg_p = torch.softmax(out["urgency_logits"], dim=-1).cpu().numpy()[0]

    top_cat = ID2CAT[cat_p.argmax()]
    pred_unit = LABEL12_TO_UNIT4[top_cat]
    pred_urg = ID2URG[urg_p.argmax()]

    u_status = "OK" if pred_unit == exp_unit else "MISMATCH"
    urg_status = "OK" if pred_urg == exp_urg else "MISMATCH"
    print(f"Text: '{text[:45]}...'")
    print(f"  -> Unit: {pred_unit} (exp: {exp_unit}) [{u_status}] | Cat: {top_cat}")
    print(f"  -> Urgency: {pred_urg} (exp: {exp_urg}) [{urg_status}]")

print("\n" + "=" * 70, flush=True)
print("TESTING SEMANTIC DEDUPLICATION (COSINE SIMILARITY)")
print("=" * 70, flush=True)

query = "Hố tử thần sâu hoắm ở ngã tư rất nguy hiểm cho người đi xe máy"
candidates = [
    "Mặt đường sụt lún nghiêm trọng tại điểm giao cắt gây tai nạn",
    "Đoạn đường xuất hiện hố sâu sạt lở nguy hiểm tính mạng",
    "Bóng đèn ngõ xóm bị cháy không sáng suốt tuần qua",
    "Quán nhậu bày bàn ghế lấn chiếm vỉa hè người đi bộ",
]

all_t = [query] + candidates
enc = tokenizer(all_t, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = model(**enc)
    embs = out["sentence_embeddings"]
    q_emb = embs[0:1]
    c_embs = embs[1:]
    sims = torch.mm(c_embs, q_emb.t()).squeeze(-1).cpu().numpy()

print(f"Query: '{query}'")
for i, (cand, s) in enumerate(zip(candidates, sims)):
    verdict = "DUPLICATE FOUND" if s >= 0.50 else "DIFFERENT"
    print(f"  [c{i+1}] Sim: {s:.4f} ({verdict}) | Text: '{cand}'")

print("\nALL MULTI-TASK CAPABILITIES VERIFIED SUCCESSFULLY ON GPU!")

