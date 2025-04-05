import json
from collections import defaultdict, Counter
from sklearn.metrics import f1_score, precision_score, recall_score

def exact_match(pred, gold):
    return int(pred == gold)

def binary_match(pred, gold):
    # Simulated GPT-4o semantic check, replace with actual GPT call in practice
    return int(pred.lower() == gold.lower())

def proportional_overlap(pred, gold):
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    overlap = pred_tokens & gold_tokens
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gold_tokens)
    return precision, recall

def evaluate_sextuples(pred_data, gold_data):
    elem_metrics = defaultdict(list)
    pair_correct, pair_pred_total, pair_gold_total = 0, 0, 0
    sextuple_correct, sextuple_pred_total, sextuple_gold_total = 0, 0, 0
    id_correct, id_total_pred, id_total_gold = 0, 0, 0

    for pred_item, gold_item in zip(pred_data, gold_data):
        pred_hexas = pred_item["hexatuple"]
        gold_hexas = gold_item["hexatuple"]

        matched_sextuples = set()
        matched_ids = set()

        for g in gold_hexas:
            g_key = (
                g["holder"]["value"], g["target"]["value"], g["aspect"]["value"],
                g["opinion"]["value"], g["sentiment"], g["rationale"]["value"]
            )
            g_key_no_sent = (
                g["holder"]["value"], g["target"]["value"], g["aspect"]["value"],
                g["opinion"]["value"], g["rationale"]["value"]
            )
            matched = False
            matched_no_sent = False

            for p in pred_hexas:
                p_key = (
                    p["holder"]["value"], p["target"]["value"], p["aspect"]["value"],
                    p["opinion"]["value"], p["sentiment"], p["rationale"]["value"]
                )
                p_key_no_sent = (
                    p["holder"]["value"], p["target"]["value"], p["aspect"]["value"],
                    p["opinion"]["value"], p["rationale"]["value"]
                )

                # Element-wise matching
                for elem in ["holder", "target", "aspect", "opinion"]:
                    score = exact_match(p[elem]["value"], g[elem]["value"])
                    elem_metrics[elem].append(score)

                # Sentiment match
                sent_score = exact_match(p["sentiment"], g["sentiment"])
                elem_metrics["sentiment"].append(sent_score)

                # Rationale: proportional F1
                pp, pr = proportional_overlap(p["rationale"]["value"], g["rationale"]["value"])
                if pp + pr > 0:
                    pf1 = 2 * pp * pr / (pp + pr)
                else:
                    pf1 = 0.0
                elem_metrics["rationale"].append(pf1)

                if p_key == g_key:
                    matched_sextuples.add(p_key)
                    matched = True
                if p_key_no_sent == g_key_no_sent:
                    matched_ids.add(p_key_no_sent)
                    matched_no_sent = True

            if not matched:
                for elem in ["holder", "target", "aspect", "opinion"]:
                    elem_metrics[elem].append(0)
                elem_metrics["sentiment"].append(0)
                elem_metrics["rationale"].append(0.0)

        # Pair & sextuple stats
        pair_pred_total += len(pred_hexas)
        pair_gold_total += len(gold_hexas)
        pair_correct += len(matched_ids)

        sextuple_pred_total += len(pred_hexas)
        sextuple_gold_total += len(gold_hexas)
        sextuple_correct += len(matched_sextuples)

        id_total_pred += len(pred_hexas)
        id_total_gold += len(gold_hexas)
        id_correct += len(matched_ids)

    # Compute final metrics
    results = {}
    for elem, scores in elem_metrics.items():
        precision = sum(scores) / len(scores) if scores else 0
        recall = sum(scores) / len(scores) if scores else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[f"{elem}_f1"] = f1

    # Pairwise
    pair_precision = pair_correct / pair_pred_total if pair_pred_total else 0
    pair_recall = pair_correct / pair_gold_total if pair_gold_total else 0
    pair_f1 = 2 * pair_precision * pair_recall / (pair_precision + pair_recall) if (pair_precision + pair_recall) else 0
    results["pairwise_f1"] = pair_f1

    # Sextuple micro F1
    micro_precision = sextuple_correct / sextuple_pred_total if sextuple_pred_total else 0
    micro_recall = sextuple_correct / sextuple_gold_total if sextuple_gold_total else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0
    results["micro_f1"] = micro_f1

    # Identification F1 (ignore sentiment)
    id_precision = id_correct / id_total_pred if id_total_pred else 0
    id_recall = id_correct / id_total_gold if id_total_gold else 0
    id_f1 = 2 * id_precision * id_recall / (id_precision + id_recall) if (id_precision + id_recall) else 0
    results["identification_f1"] = id_f1

    return results