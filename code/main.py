import os
import json
from collections import defaultdict
from conllu import parse_incr
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- Parsing Functions ---
def parse_conllu(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for tokenlist in parse_incr(f):
            sentence = [(token['id'], token['form'], token['head']) for token in tokenlist if isinstance(token['id'], int)]
            sentences.append(sentence)
    return sentences

# --- Alignment and Filtering ---
def align_and_filter_sentences(parser_outputs, gold_output):
    num_parsers = len(parser_outputs)
    num_sents = [len(p) for p in parser_outputs] + [len(gold_output)]
    min_sents = min(num_sents)

    aligned_indices = []
    for sid in range(min_sents):
        tokens_list = [[w[1] for w in p[sid]] for p in parser_outputs]
        gold_tokens = [w[1] for w in gold_output[sid]]
        ref_tokens = tokens_list[0]
        if all(t == ref_tokens for t in tokens_list) and ref_tokens == gold_tokens:
            aligned_indices.append(sid)

    filtered_parsers = [[p[sid] for sid in aligned_indices] for p in parser_outputs]
    filtered_gold = [gold_output[sid] for sid in aligned_indices]

    return filtered_parsers, filtered_gold

# --- Label Matrix ---
def build_label_matrix(sentences_list):
    label_matrices = []
    for sent_idx in range(len(sentences_list[0])):
        edges = set()
        for p_out in sentences_list:
            edges.update((h, d) for d, _, h in p_out[sent_idx] if h != 0)
        edge_list = list(edges)
        mat = np.zeros((len(edge_list), len(sentences_list)))
        for p_idx, p_out in enumerate(sentences_list):
            pred_edges = set((h, d) for d, _, h in p_out[sent_idx] if h != 0)
            for e_idx, e in enumerate(edge_list):
                mat[e_idx, p_idx] = 1 if e in pred_edges else -1
        label_matrices.append((edge_list, mat))
    return label_matrices

# --- Aggregation Methods ---
def aggregate_mst(edge_list, mat):
    votes = np.sum(mat, axis=1)
    G = nx.DiGraph()
    for idx, (h, d) in enumerate(edge_list):
        G.add_edge(h, d, weight=votes[idx])
    return list(nx.maximum_spanning_arborescence(G).edges())

def aggregate_crh(edge_list, mat, max_iter=10):
    m = mat.shape[1]
    w = np.ones(m) / m
    y = np.sign(np.dot(mat, w))
    for _ in range(max_iter):
        # Update weights based on disagreement (0-1 loss as distance metric)
        for k in range(m):
            disagreements = (y != mat[:, k]).astype(float)
            w[k] = np.exp(-np.sum(disagreements))
        w /= np.sum(w)
        # Update aggregated labels using weighted majority vote
        y = np.sign(np.dot(mat, w))
        y[y == 0] = 1  # break ties in favor of +1
    selected = [edge_list[i] for i, val in enumerate(y) if val > 0]
    return selected

def aggregate_cim(edge_list, mat):
    m = mat.shape[1]
    parser_reliability = np.zeros(m)
    for j in range(m):
        X = np.delete(mat, j, axis=1)
        y = mat[:, j]
        clf = LogisticRegression(penalty='l1', solver='liblinear')
        clf.fit(X, y)
        coef = clf.coef_[0]
        parser_reliability[j] = np.sum(np.abs(coef))

    reliability_weights = parser_reliability / np.sum(parser_reliability) if np.sum(parser_reliability) > 0 else np.ones(m)/m
    prob_scores = (mat @ reliability_weights + 1) / 2  # Normalize to [0,1]

    G = nx.DiGraph()
    for idx, (h, d) in enumerate(edge_list):
        G.add_edge(h, d, weight=prob_scores[idx])
    return list(nx.maximum_spanning_arborescence(G).edges())

# --- Evaluation ---
def compute_uas(predicted_trees, gold_sentences):
    total, correct = 0, 0
    for sid, pred_edges in predicted_trees.items():
        gold_edges = set((h, d) for d, _, h in gold_sentences[sid] if h != 0)
        total += len(gold_edges)
        correct += len(set(pred_edges) & gold_edges)
    return correct / total if total > 0 else 0

# --- Main ---
data_dir = "../data/official-submissions"
gold_file = "../data/00-gold-standard/en_ewt.conllu"
baselines = ["Stanford-18", "HIT-SCIR-18"]

parser_outputs = [parse_conllu(os.path.join(data_dir, b, "en_ewt.conllu")) for b in baselines]
gold_output = parse_conllu(gold_file)

parser_outputs, gold_output = align_and_filter_sentences(parser_outputs, gold_output)
label_matrices = build_label_matrix(parser_outputs)

results = {"MST": {}, "CRH": {}, "CIM": {}}

for method in results.keys():
    aggregated = {}
    for sid, (edge_list, mat) in enumerate(label_matrices):
        if method == "MST":
            edges = aggregate_mst(edge_list, mat)
        elif method == "CRH":
            edges = aggregate_crh(edge_list, mat)
        elif method == "CIM":
            edges = aggregate_cim(edge_list, mat)
        aggregated[sid] = edges
    uas = compute_uas(aggregated, gold_output)
    results[method]['UAS'] = uas

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Evaluation complete. Results saved to evaluation_results.json.")
