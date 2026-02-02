O-MRE: Data and Model Code
=========================

Overlapping Multimodal Relation Extraction (O-MRE)  
(O-MNRE, O-JMERE)

This repository provides the datasets and model implementation for the paper:

**“HIN: A Unified Framework for Knowledge-Dense Overlapping Multimodal Relation Extraction”**  
(*****, 2026)

The repository supports both:
- construction and usage of overlapping multimodal relation extraction benchmarks, and  
- training and evaluation of the proposed **HIN** model.

------------------------------------------------------------------
1. Dataset Description
------------------------------------------------------------------

The O-MRE datasets are designed to evaluate **Overlapping Multimodal Relation Extraction (O-MRE)** models that integrate textual and visual information.

Each instance contains:
- a sentence with annotated entities,
- multiple relation triples (possibly overlapping),
- and corresponding image references from the original multimodal datasets.

The data is provided in JSON format (e.g., `merged_test_data.json`), where each record corresponds to one multimodal instance.

------------------------------------------------------------------
2. JSON Field Descriptions
------------------------------------------------------------------

Field             | Type   | Description
------------------|--------|-----------------------------------------------------
words             | list   | Tokenized text sequence; each element represents one token.
entity_list       | list   | Entity annotations with `name` and `pos` (token indices).
entity_pair_list  | list   | Relation triples: [head, tail, relation_id, sample_id].
imgids            | list   | Original image filenames or paths.
aux_imgs          | list   | Auxiliary image crops (e.g., YOLO-based).
rcnn_imgs         | list   | Region-cropped images from Faster R-CNN.

------------------------------------------------------------------
3. Image Data Source
------------------------------------------------------------------

Image references are derived from the official releases of:

- MNRE: https://github.com/thecharm/MNRE  
- JMERE: https://github.com/jmre-team/JMERE  

This repository does **not** redistribute image files.


Unzip the data and rename the directory as `mnre`, `jmere`, which should be placed in the directory `code/data`:

```bash
cd code
mkdir data logs ckpt
```

------------------------------------------------------------------
4. Automatic Construction of O-MRE Dataset
------------------------------------------------------------------

Run the provided conversion script:

```bash
python convert.py -t mnre
python convert.py -t jmere
```

------------------------------------------------------------------
5. Model: HIN
------------------------------------------------------------------

We provide the official PyTorch implementation of **HIN (Hierarchical Interaction Network)**,
which addresses knowledge-dense overlapping relation extraction via:

- hierarchical cross-modal interaction,
- gated bottleneck reweighting,
- and a smoothing decoder.

```bash
cd code
conda create -n hin python=3.9
conda activate hin
pip install -r requirements.txt
```

Training the model

The best hyperparameters we found have been witten in `run_mre_overlap_matrix.sh` and `run_jmere_overlap_matrix.sh`file.

You can simply run the bash script for multimodal relation extraction:

```bash
bash run_****_overlap_matrix.sh
```

------------------------------------------------------------------
6. Citation
------------------------------------------------------------------

```bibtex
@article{wang2026hin,
  title={HIN: A Unified Framework for Knowledge-Dense Overlapping Multimodal Relation Extraction},
  author={Wang, Hailin and Ren, Hangyi and Zhang, Dan and Jianzhang, Wei and Du, Zhekai and Liu, Guisong and Qin, Ke},
  journal={***********},
  year={2026}
}
```
