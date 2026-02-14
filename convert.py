#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert.py

Usage:
  python convert.py -t mnre
  python convert.py -t jmere

This script converts the original MNRE / JMERE-style "single-triple-per-sample" data
into an "overlapping RE" merged dataset:
  o-<datasetname>/merged_train_data.json
  o-<datasetname>/merged_dev_data.json
  o-<datasetname>/merged_test_data.json

Folder assumptions (same as the provided ipynb):
  data/
    mnre/
      txt/...
      img_detect/...
      img_org/...
      img_vg/...
      ours_rel2id.json
    jmere/
      txt/...
      img_detect/...
      img_org/...
      img_vg/...
      ours_rel2id.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch


def build_paths(data_root: str) -> Dict[str, Dict[str, str]]:
    """Build DATA_PATH dict (copied from the ipynb, but rooted at data_root)."""
    return {
        "mnre": {
            "train": os.path.join(data_root, "mnre", "txt", "ours_train.txt"),
            "dev": os.path.join(data_root, "mnre", "txt", "ours_val.txt"),
            "test": os.path.join(data_root, "mnre", "txt", "ours_test_s.txt"),
            "train_auximgs": os.path.join(data_root, "mnre", "txt", "mre_train_dict.pth"),
            "dev_auximgs": os.path.join(data_root, "mnre", "txt", "mre_dev_dict.pth"),
            "test_auximgs": os.path.join(data_root, "mnre", "txt", "mre_test_dict_s.pth"),
            "train_img2crop": os.path.join(data_root, "mnre", "img_detect", "train", "train_img2crop.pth"),
            "dev_img2crop": os.path.join(data_root, "mnre", "img_detect", "val", "val_img2crop.pth"),
            "test_img2crop": os.path.join(data_root, "mnre", "img_detect", "test", "test_img2crop_s.pth"),
        },
        "jmere": {
            "train": os.path.join(data_root, "jmere", "txt", "v1_ours_train.json"),
            "dev": os.path.join(data_root, "jmere", "txt", "v1_ours_val.json"),
            "test": os.path.join(data_root, "jmere", "txt", "v1_ours_test.json"),
            "train_auximgs": os.path.join(data_root, "jmere", "txt", "mre_train_dict.pth"),
            "dev_auximgs": os.path.join(data_root, "jmere", "txt", "mre_dev_dict.pth"),
            "test_auximgs": os.path.join(data_root, "jmere", "txt", "mre_test_dict.pth"),
            "train_img2crop": os.path.join(data_root, "jmere", "img_detect", "train", "train_img2crop.pth"),
            "dev_img2crop": os.path.join(data_root, "jmere", "img_detect", "val", "val_img2crop.pth"),
            "test_img2crop": os.path.join(data_root, "jmere", "img_detect", "test", "test_img2crop.pth"),
        },
    }


def get_relation_dict(data_root: str, dataset: str) -> Dict[str, int]:
    """Read ours_rel2id.json (same as ipynb get_relation_dict)."""
    re_path = os.path.join(data_root, dataset, "ours_rel2id.json")
    with open(re_path, "r", encoding="utf-8") as f:
        line = f.readlines()[0]
        return json.loads(line)


def _safe_load_list_from_file(path: str) -> Any:
    """JMERE txt json sometimes stored as a python-literal list in a single line."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    # Try JSON first, then python literal.
    try:
        return json.loads(raw)
    except Exception:
        return ast.literal_eval(raw)


def load_ori_data(data_root: str, dataset: str, mode: str) -> Dict[str, Any]:
    """
    Load original (single-triple) data into a unified dict:
      words, relations, heads, tails, imgids, dataid, aux_imgs, rcnn_imgs

    This follows the ipynb logic:
      - MNRE: each line is a python dict string with keys: token, relation, h, t, img_id
      - JMERE: file is a list; each item has token, img_id, label_list; each label has:
               relation, beg_ent{name,pos}, sec_ent{name,pos}
    """
    paths = build_paths(data_root)
    if dataset not in paths:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected one of: {list(paths.keys())}")

    file_path = paths[dataset][mode]
    aux_path = paths[dataset][f"{mode}_auximgs"]
    img2crop_path = paths[dataset][f"{mode}_img2crop"]

    words: List[List[str]] = []
    relations: List[str] = []
    heads: List[Dict[str, Any]] = []
    tails: List[Dict[str, Any]] = []
    imgids: List[str] = []
    dataid: List[int] = []

    if dataset == "mnre":
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            ex = ast.literal_eval(line)  # same as ipynb
            words.append(ex["token"])
            relations.append(ex["relation"])
            heads.append(ex["h"])  # {name, pos}
            tails.append(ex["t"])  # {name, pos}
            imgids.append(ex["img_id"])
            dataid.append(i)

    elif dataset == "jmere":
        lines = _safe_load_list_from_file(file_path)
        # ipynb expands each label into a separate triple, but keeps dataid = i (original line index)
        for i, ex in enumerate(lines):
            for label in ex["label_list"]:
                words.append(ex["token"])
                imgids.append(ex["img_id"])
                relations.append(label[0]["relation"])
                head = {"name": label[0]["beg_ent"]["name"], "pos": label[0]["beg_ent"]["pos"]}
                tail = {"name": label[0]["sec_ent"]["name"], "pos": label[0]["sec_ent"]["pos"]}
                heads.append(head)
                tails.append(tail)
                dataid.append(i)

    aux_imgs = torch.load(aux_path) if os.path.exists(aux_path) else {}
    rcnn_imgs = torch.load(img2crop_path) if os.path.exists(img2crop_path) else {}

    assert len(words) == len(relations) == len(heads) == len(tails) == len(imgids) == len(dataid)

    return {
        "words": words,
        "relations": relations,
        "heads": heads,
        "tails": tails,
        "imgids": imgids,
        "dataid": dataid,
        "aux_imgs": aux_imgs,
        "rcnn_imgs": rcnn_imgs,
    }


def merge_samples_and_build_entity_lists(data: Dict[str, Any], rel2id: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Merge duplicated sentence samples (same " ".join(words)) into one instance, and build:
      - entity_list: list of {name, pos}
      - entity_pair_list: list of [h_id, t_id, rel_id, dataid]
    Also merges:
      - imgids (list)
      - aux_imgs (list)
      - rcnn_imgs (list)

    NOTE: This is faithful to the ipynb: dedup key is sentence only (not (sentence,imgid)).
          The instance keeps a list of imgids.
    """
    sentence_idx: Dict[int, int] = {}
    merged_data: List[Dict[str, Any]] = []

    for words, relation, head, tail, imgid, did in zip(
        data["words"], data["relations"], data["heads"], data["tails"], data["imgids"], data["dataid"]
    ):
        imgid_key = os.path.splitext(imgid)[0]
        sentence_key = " ".join(words)
        sentence_hash = hash(sentence_key)

        if sentence_hash in sentence_idx:
            idx = sentence_idx[sentence_hash]
            existing = merged_data[idx]

            head_entity = {"name": head["name"], "pos": head["pos"]}
            tail_entity = {"name": tail["name"], "pos": tail["pos"]}

            # entity id lookup (exact match)
            head_id = next((i for i, e in enumerate(existing["entity_list"]) if e == head_entity), None)
            if head_id is None:
                head_id = len(existing["entity_list"])
                existing["entity_list"].append(head_entity)

            tail_id = next((i for i, e in enumerate(existing["entity_list"]) if e == tail_entity), None)
            if tail_id is None:
                tail_id = len(existing["entity_list"])
                existing["entity_list"].append(tail_entity)

            rel_id = rel2id[relation]
            new_pair = [head_id, tail_id, rel_id, did]

            # avoid exact duplicate (same h,t,rel); allow same h,t with different rel (multi-label)
            exists_same = False
            for p in existing["entity_pair_list"]:
                if p[0] == head_id and p[1] == tail_id and p[2] == rel_id:
                    exists_same = True
                    break
            if not exists_same:
                existing["entity_pair_list"].append(new_pair)

            # merge imgids
            if imgid not in existing["imgids"]:
                existing["imgids"].append(imgid)

            # merge aux_imgs
            aux_list = data["aux_imgs"].get(did, [])
            if aux_list:
                # aux_list can be list/tuple; ensure iterable
                for a in aux_list:
                    if a not in existing["aux_imgs"]:
                        existing["aux_imgs"].append(a)

            # merge rcnn_imgs
            if imgid_key in data["rcnn_imgs"]:
                for crop in data["rcnn_imgs"][imgid_key]:
                    if crop not in existing["rcnn_imgs"]:
                        existing["rcnn_imgs"].append(crop)

        else:
            head_entity = {"name": head["name"], "pos": head["pos"]}
            tail_entity = {"name": tail["name"], "pos": tail["pos"]}

            new_sample = {
                "words": words,
                "entity_list": [head_entity, tail_entity],
                "entity_pair_list": [[0, 1, rel2id[relation], did]],
                "imgids": [imgid],
                "aux_imgs": list(data["aux_imgs"].get(did, [])),
                "rcnn_imgs": list(data["rcnn_imgs"].get(imgid_key, [])),
            }
            sentence_idx[sentence_hash] = len(merged_data)
            merged_data.append(new_sample)

    return merged_data


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--target",
        required=True,
        choices=["mnre", "jmere"],
        help="Dataset name to convert: mnre or jmere",
    )
    parser.add_argument(
        "--data_root",
        default="code/data",
        help="Root folder that contains mnre/ and jmere/ (default: ./data)",
    )
    parser.add_argument(
        "--out_root",
        default=None,
        help="Output root (default: <data_root>/o-<dataset>)",
    )
    args = parser.parse_args()

    dataset = args.target
    data_root = args.data_root
    out_root = args.out_root or os.path.join(data_root, f"o-{dataset}")
    os.makedirs(out_root, exist_ok=True)

    rel2id = get_relation_dict(data_root, dataset)

    for split in ["train", "dev", "test"]:
        print(f"[{dataset}] Loading {split}...")
        data = load_ori_data(data_root, dataset, split)

        print(f"[{dataset}] Merging {split}...")
        merged = merge_samples_and_build_entity_lists(data, rel2id)

        out_path = os.path.join(out_root, f"merged_{split}_data.json")
        save_json(merged, out_path)
        print(f"[{dataset}] Saved: {out_path}  (#instances={len(merged)})")

    print(f"Done. Output folder: {out_root}")


if __name__ == "__main__":
    main()
