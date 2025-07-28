import torch
import os 
import json


def save_embeddings(embeddings, id_list, output_path):
    assert len(embeddings) == len(id_list)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    torch.save({
        "embeddings": embeddings,
        "ids": id_list
    }, output_path)


def save_mapping(sequences_ids, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    seq_to_idx = {seq: i for i, seq in enumerate(sequences_ids)}
    map_path = os.path.splitext(output_path)[0] + "_seq_to_idx.json"
    with open(map_path, 'w') as f:
        json.dump(seq_to_idx, f)
    print(f"Saved sequence to index mapping to {map_path}")