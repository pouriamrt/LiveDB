import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import os
from loguru import logger
import asyncio
from typing import List, Tuple
from Config import config

# ----- same definitions as before -----
LABELS = ["no", "maybe", "yes"]
TASKS = ["P_AB", "I_AB", "C_AB", "O_AB", "S_AB"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


class MultiHeadClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_tasks: int, num_classes: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        # one head per task
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, num_classes) for _ in range(num_tasks)]
        )

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        cls = self.dropout(out[:, 0, :])  # [CLS]-like first token for DistilBERT
        logits = [head(cls) for head in self.heads]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for t_idx in range(len(self.heads)):
                losses.append(loss_fct(logits[t_idx], labels[:, t_idx]))
            loss = sum(losses) / len(losses)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


# ----- reload -----
LOAD_DIR = config.MODEL_DIR
tokenizer = AutoTokenizer.from_pretrained(LOAD_DIR)
base_model_name = LOAD_DIR

model = MultiHeadClassifier(
    base_model_name, num_tasks=len(TASKS), num_classes=len(LABELS)
)
state_dict = torch.load(
    os.path.join(LOAD_DIR, "classifier_heads.pt"), map_location="cpu"
)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()
logger.info("Model loaded and ready.")


def check_abs_model(
    text: str, max_length: int = 512, device: str = "cpu"
) -> Tuple[List[List[int]], List[List[float]]]:
    enc = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
    )

    all_preds = [[] for _ in TASKS]
    all_confs = [[] for _ in TASKS]

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        logits_list = out["logits"] if isinstance(out, dict) else out
        probs_list = [torch.softmax(logits, dim=-1) for logits in logits_list]

        for t_idx in range(len(TASKS)):
            preds = probs_list[t_idx].argmax(dim=-1).tolist()
            confs = probs_list[t_idx].max(dim=-1).values.tolist()
            all_preds[t_idx].extend(preds)
            all_confs[t_idx].extend(confs)

        del enc, out, logits_list, probs_list
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    out_cols_labels = {
        f"{TASKS[t]}_pred": [LABELS[i] for i in all_preds[t]] for t in range(len(TASKS))
    }
    out_cols_confs = {f"{TASKS[t]}_conf": all_confs[t] for t in range(len(TASKS))}

    return out_cols_labels, out_cols_confs


async def check_abs_model_async(
    text: str, max_length: int = 512, device: str = "cpu"
) -> Tuple[List[List[int]], List[List[float]]]:
    return await asyncio.to_thread(check_abs_model, text, max_length, device)


async def main():
    test_text = """
    Digital Intervention Promoting Physical Activity in People Newly Diagnosed with Parkinson’s Disease: 
    Feasibility and Acceptability of the Knowledge, Exercise-Efficacy and Participation (KEEP) Intervention
    Background:
    Exercise promotion interventions for people with Parkinson’s disease (PD) are often offered on a face-to-face basis, follow a generic 
    “one-size-fit-all” approach, and are not typically delivered at diagnosis. Considering PD’s heterogenous nature, the existing 
    evidence on the merits of exercise on symptom management and the expressed wishes of people living with PD for access to timely and 
    tailored evidence-based information, there is a demand for interventions that are easily accessible, scalable and co-designed with 
    people living with PD.
    Objective:
    Evaluate the feasibility and acceptability of a co-designed digital intervention promoting exercise and physical activity, in people 
    newly diagnosed with PD.
    Methods:
    Thirty people living with PD for less than one year participated in an assessor-blinded randomized feasibility trial from June 2022 to 
    April 2023. The intervention group received the 8-week Knowledge, Exercise Efficacy and Participation (KEEP) intervention comprising 6 
    interactive digital modules and 4 online live group discussions facilitated by a specialist physiotherapist. Assessments were performed 
    at baseline, post intervention and at 6-month follow up.
    Results:
    Thirty participants were recruited to target with a (64%) recruitment rate (30/47). All but one participant completed the 6-month follow-up 
    assessment. There was high retention (97%), module completion (91%), and online discussion attendance (88%). Outcome measure collection 
    was feasible, including accelerometer data with a daily average wear time of 23.9 hours (SD:0.295).
    Conclusions:
    The KEEP intervention was feasible and acceptable in people newly diagnosed with PD. A larger trial is needed to assess intervention efficacy 
    and correlation between knowledge, self-efficacy, and activity levels.
    """
    preds, confs = await check_abs_model_async(test_text)
    print(preds, confs)


if __name__ == "__main__":
    asyncio.run(main())
