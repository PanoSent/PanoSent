# evaluate_task1.py
import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="")

def is_semantically_equivalent(pred_term: str, gold_term: str) -> bool:
    prompt = (
        f"Do the two terms '{pred_term}' and '{gold_term}' have similar meanings? Don't need to be strict.\n"
        "Answer with only 'Yes' or 'No'."
    )
    for _ in range(10):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            reply = response.choices[0].message.content.strip().lower()
            return reply.startswith("yes")
        except Exception as e:
            print(f"[GPT ERROR - retrying]: {e}")
            time.sleep(1)
    return False

def match_explicit_or_implicit(pred_field: dict, gold_field: dict) -> bool:
    pred_val = pred_field["value"].strip()
    gold_val = gold_field["value"].strip()
    if gold_field.get("manner", "explicit") == "explicit":
        return pred_val == gold_val
    return is_semantically_equivalent(pred_val, gold_val)

def proportional_overlap(pred: str, gold: str) -> float:
    pred_set = set(pred.lower().split())
    gold_set = set(gold.lower().split())
    intersection = pred_set & gold_set
    if not pred_set or not gold_set:
        return 0.0
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def match_sextuple(pred_sxt: dict, gold_sxt: dict, ignore_sentiment=False, rationale_threshold=0.5) -> bool:
    for key in ["holder", "target", "aspect", "opinion"]:
        if not match_explicit_or_implicit(pred_sxt[key], gold_sxt[key]):
            return False
    if not ignore_sentiment:
        if pred_sxt["sentiment"].strip().lower() != gold_sxt["sentiment"].strip().lower():
            return False
    return proportional_overlap(pred_sxt["rationale"]["value"], gold_sxt["rationale"]["value"]) >= rationale_threshold

def load_raw_json(path):
    with open(path) as f:
        return {entry["doc_id"]: entry for entry in json.load(f)}

def evaluate_sextuple(pred_path, gold_path):
    preds = load_raw_json(pred_path)
    refs = load_raw_json(gold_path)

    matched_micro = matched_iden = 0
    total_pred = total_gold = 0
    rationale_threshold = 0.5

    for doc_id in tqdm(refs, desc="Evaluating Task 1"):
        pred_sxts = preds.get(doc_id, {}).get("hexatuple", [])
        gold_sxts = refs[doc_id]["hexatuple"]

        used = [False] * len(gold_sxts)
        used_iden = [False] * len(gold_sxts)

        for p in pred_sxts:
            total_pred += 1
            for i, g in enumerate(gold_sxts):
                if not used[i] and match_sextuple(p, g, ignore_sentiment=False, rationale_threshold=rationale_threshold):
                    matched_micro += 1
                    used[i] = True
                    break

        for p in pred_sxts:
            for i, g in enumerate(gold_sxts):
                if not used_iden[i] and match_sextuple(p, g, ignore_sentiment=True, rationale_threshold=rationale_threshold):
                    matched_iden += 1
                    used_iden[i] = True
                    break

        total_gold += len(gold_sxts)

    micro_p = matched_micro / total_pred if total_pred else 0
    micro_r = matched_micro / total_gold if total_gold else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0

    iden_p = matched_iden / total_pred if total_pred else 0
    iden_r = matched_iden / total_gold if total_gold else 0
    iden_f1 = 2 * iden_p * iden_r / (iden_p + iden_r) if iden_p + iden_r else 0

    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Identification F1: {iden_f1:.4f}")
    print(f"Final Score: {(micro_f1 + iden_f1) / 2:.4f}")

if __name__ == "__main__":
    evaluate_sextuple(".json", "task1_reference.json")