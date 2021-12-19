import pathlib
import gdown
import argparse
import torch
import smbop
from allennlp.models.archival import Archive, load_archive, archive_model
from allennlp.data.vocabulary import Vocabulary
from smbop.modules.relation_transformer import *
from allennlp.common import Params
from smbop.models.smbop import SmbopParser
from smbop.modules.lxmert import LxmertCrossAttentionLayer
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
import itertools
import smbop.utils.node_util as node_util
import numpy as np
from allennlp.models import Model
from allennlp.common.params import *
from allennlp.data import DatasetReader, Instance
import tqdm
from allennlp.predictors import Predictor
import json
import importlib
from apted import APTED, Config
# from zss import simple_distance, Node
from binarytree import tree
from time import time
from tqdm import tqdm
import anytree
import namegenerator
import pickle
from multiprocessing import Pool

import signal
import pdb



compset=set()
class CustomConfig(Config):
    def __init__(self):
        super().__init__()
        # all_names={'union', 'Orderby_desc', 'gt', 'lte', 'Table', 'Project', 'Selection', 'eq', 'sum', 'Subquery', 'Or', 'keep', 'Val_list', 'distinct', 'like', 'in', 'Value', 'lt', 'literal', 'intersect', 'Limit', 'except', 'neq', 'And', 'gte', 'max', 'Groupby', 'avg', 'count', 'min', 'nin', 'Orderby_asc', 'Product'}

        self.agg_grp=['max','min','avg','count','sum']
        self.order_grp=['Orderby_desc','Orderby_asc']
        self.boolean_grp=['Or','And']
        self.set_grp=['union','intersect','except']
        self.leaf_grp=['Val_list','Value','literal','Table']
        self.sim_grp=['like','in','nin']
        self.comp_grp=['gt','lte','eq','lt','gte','neq']

    def rename(self, node1, node2):
        
        if (node1.name in self.agg_grp and node2.name in self.agg_grp) or \
        (node1.name in self.order_grp and node2.name in self.order_grp) or \
        (node1.name in self.boolean_grp and node2.name in self.boolean_grp) or \
        (node1.name in self.set_grp and node2.name in self.set_grp) or \
        (node1.name in self.leaf_grp and node2.name in self.leaf_grp) or \
        (node1.name in self.sim_grp and node2.name in self.sim_grp) or \
        (node1.name in self.comp_grp and node2.name in self.comp_grp):
            return 1 if node1.name != node2.name else 0
        else:
            return 2 if node1.name != node2.name else 0
    def children(self, node):
        return [x for x in node.children]

def to_string(value):
    if isinstance(value, list):
        return [to_string(x) for x in value]
    elif isinstance(value, bool):
        return "true" if value else "false"
    else:
        return str(value)

def dist(x,y):
    '''  x and y are instance objects whose distance we need'''
    ccobj=CustomConfig()
    apted = APTED(x['tree_obj'].metadata, y['tree_obj'].metadata,ccobj)
    return (apted.compute_edit_distance())




def run():
    importlib.reload(smbop)
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--name", nargs="?")
    parser.add_argument("--force", action="store_true",
                        help="""If True, we will overwrite the serialization
                                directory if it already exists.""")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--recover", action="store_true",
                        help= """If True, we will try to recover a training run
                                 from an existing serialization directory. 
                                 This is only intended for use when something 
                                 actually crashed during the middle of a run. 
                                 For continuing training a model on new data,
                                  see Model.from_archive.""")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--detect_anomoly", action="store_true") #IDK
    parser.add_argument("--profile", action="store_true") #IDK: some sort of debugging funct
    parser.add_argument("--is_oracle", action="store_true")
    parser.add_argument("--tiny_dataset", action="store_true")
    parser.add_argument("--load_less", action="store_true")
    parser.add_argument("--cntx_rep", action="store_true")
    parser.add_argument("--cntx_beam", action="store_true")
    parser.add_argument("--disable_disentangle_cntx", action="store_true")
    parser.add_argument("--disable_cntx_reranker", action="store_true")
    parser.add_argument("--disable_value_pred", action="store_true")
    parser.add_argument("--disable_use_longdb", action="store_true")
    parser.add_argument("--uniquify", action="store_true")
    parser.add_argument("--use_bce", action="store_true")
    parser.add_argument("--tfixup", action="store_true")
    parser.add_argument("--train_as_dev", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--disable_utt_aug", action="store_true")
    parser.add_argument("--should_rerank", action="store_true")
    parser.add_argument("--use_treelstm", action="store_true")
    parser.add_argument("--disable_db_content", action="store_true",
                        help="Run with this argument (once) before pre-proccessing to reduce the pre-proccessing time by half \
                         This argument causes EncPreproc to not perform IR on the largest tables. ")
    parser.add_argument("--lin_after_cntx", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--rat_layers", type=int, default=8)
    parser.add_argument("--beam_size", default=30, type=int)
    parser.add_argument("--base_dim", default=32, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--beam_encoder_num_layers", default=1, type=int)
    parser.add_argument("--tree_rep_transformer_num_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--rat_dropout", default=0.2, type=float)
    parser.add_argument("--lm_lr", default=3e-6, type=float)
    parser.add_argument("--lr", type=float, default=0.000186)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--grad_acum", default=4, type=int)
    parser.add_argument("--max_steps", default=60000, type=int)
    parser.add_argument("--power", default=0.5, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--grad_clip", default=-1, type=float)
    parser.add_argument("--grad_norm", default=-1, type=float)

    default_dict = {k.option_strings[0][2:]: k.default for k in parser._actions}
    args = parser.parse_args()
    diff = "_".join(
        [
            f"{key}{value}"
            for key, value in vars(args).items()
            if (key != "name" and value != default_dict[key])
        ]
    ) #vars which differ from default

    ext_vars = {}
    for key, value in vars(args).items():
        if key.startswith("disable"):
            new_key = key.replace("disable_", "")
            ext_vars[new_key] = to_string(not value)
        else:
            ext_vars[key] = to_string(value)
    print(ext_vars) #just a toggle of disabled variables
    #default_config_file = "configs/trial_extract.jsonnet"
    default_config_file = "configs/train_extract.jsonnet"

    overrides_dict = {}

    if args.profile:
        overrides_dict["trainer"]["num_epochs"] = 1

    experiment_name_parts = []
    experiment_name_parts.append(namegenerator.gen())
    if diff:
        experiment_name_parts.append(diff)
    if args.name:
        experiment_name_parts.append(args.name)

    experiment_name = "_".join(experiment_name_parts)
    print(f"experiment_name: {experiment_name}")
    ext_vars["experiment_name"] = experiment_name
    overrides_json = json.dumps(overrides_dict)
    settings = Params.from_file(
        default_config_file,
        # ext_vars=ext_vars,
        # params_overrides=overrides_json,
    )
    dbr=SmbopSpiderDatasetReader.from_params(settings)
    tlist=[]
    # ques=[]
    for inst in (dbr._read_examples_file("dataset/train_spider.json")):
        tlist.append(inst)
    print(len(tlist))
    print(dist(x,y))


if __name__=='__main__':
    run()