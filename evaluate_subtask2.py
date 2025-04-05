from sklearn.metrics import precision_recall_fscore_support
import json

def evaluate_flip_trigger(pred_data, gold_data):
    # Metrics for (1) Sentiment flip match, (2) Trigger classification, (3) Flip+Trigger exact match
    flip_correct, flip_total_pred, flip_total_gold = 0, 0, 0
    flip_trig_correct, flip_trig_total_pred, flip_trig_total_gold = 0, 0, 0

    trigger_gold = []
    trigger_pred = []

    for pred_item, gold_item in zip(pred_data, gold_data):
        pred_flips = pred_item.get("sentiment flip", [])
        gold_flips = gold_item.get("sentiment flip", [])

        gold_dict = {(f["holder"], f["target"], f["aspect"]): f for f in gold_flips}
        pred_dict = {(f["holder"], f["target"], f["aspect"]): f for f in pred_flips}

        for key in set(gold_dict.keys()) | set(pred_dict.keys()):
            g = gold_dict.get(key)
            p = pred_dict.get(key)

            if g and p:
                # Flip correctness
                flip_pred = (p["initial sentiment"], p["flipped sentiment"])
                flip_gold = (g["initial sentiment"], g["flipped sentiment"])
                if flip_pred == flip_gold:
                    flip_correct += 1

                flip_total_pred += 1
                flip_total_gold += 1

                # Trigger classification
                trigger_gold.append(g["trigger type"])
                trigger_pred.append(p["trigger type"])

                # Combined match
                if flip_pred == flip_gold and p["trigger type"] == g["trigger type"]:
                    flip_trig_correct += 1

                flip_trig_total_pred += 1
                flip_trig_total_gold += 1

            elif g:
                flip_total_gold += 1
                trigger_gold.append(g["trigger type"])
            elif p:
                flip_total_pred += 1
                flip_trig_total_pred += 1
                trigger_pred.append(p["trigger type"])

    # Exact Match F1 for Flip
    prec_flip = flip_correct / flip_total_pred if flip_total_pred else 0
    rec_flip = flip_correct / flip_total_gold if flip_total_gold else 0
    f1_flip = 2 * prec_flip * rec_flip / (prec_flip + rec_flip) if (prec_flip + rec_flip) else 0

    # Macro F1 for Trigger
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        trigger_gold, trigger_pred, average='macro', zero_division=0)

    # Exact Match F1 for Flip-Trig
    prec_fliptrig = flip_trig_correct / flip_trig_total_pred if flip_trig_total_pred else 0
    rec_fliptrig = flip_trig_correct / flip_trig_total_gold if flip_trig_total_gold else 0
    f1_fliptrig = 2 * prec_fliptrig * rec_fliptrig / (prec_fliptrig + rec_fliptrig) if (prec_fliptrig + rec_fliptrig) else 0

    return {
        "f1_flip": f1_flip,
        "f1_trigger_macro": f1_macro,
        "f1_flip_trig": f1_fliptrig
    }