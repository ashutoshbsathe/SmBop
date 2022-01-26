import argparse
import torch

from allennlp.models.archival import Archive, load_archive, archive_model
from allennlp.data.vocabulary import Vocabulary
from smbop.modules.relation_transformer import *
import json
from allennlp.common import Params
from smbop.models.smbop_clf import SmbopParser
from smbop.modules.lxmert import LxmertCrossAttentionLayer
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
from smbop.dataset_readers.pickle_reader import PickleReader
import smbop.dataset_readers.disamb_sql as disamb_sql
import itertools
import smbop.utils.node_util as node_util
import numpy as np
import numpy as np
import json
import tqdm
from allennlp.models import Model
from allennlp.common.params import *
from allennlp.data import DatasetReader, Instance
import tqdm
from allennlp.predictors import Predictor
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_path",type=str)
    parser.add_argument("--dev_path", type=str, default="dataset/dev.json")
    parser.add_argument("--table_path", type=str, default="dataset/tables.json")
    parser.add_argument("--dataset_path", type=str, default="dataset/database")
    parser.add_argument(
        "--output", type=str, default="predictions_with_vals_fixed4.txt"
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    overrides = {
        "dataset_reader": {
            "tables_file": args.table_path,
            "dataset_path": args.dataset_path,
        }
    }
    overrides["validation_dataset_reader"] = {
        "tables_file": args.table_path,
        "dataset_path": args.dataset_path,
    }
    predictor = Predictor.from_path(
        args.archive_path, cuda_device=args.gpu, overrides=overrides
    )
    print("after pred")
    
    outputs = []
    with open(args.output, "w") as g:
        with open(args.dev_path) as f:
            dev_json = json.load(f)
            for i, el in enumerate(tqdm.tqdm(dev_json)):
                if 'query_toks' in el:
                    try:
                        ex = disamb_sql.fix_number_value(el)
                        sql = disamb_sql.disambiguate_items(
                            ex['db_id'],
                            ex['query_toks_no_value'],
                            predictor._dataset_reader._tables_file,
                            allow_aliases=False,
                        )
                        sql_with_values = disamb_sql.sanitize(ex['query'])
                    except Exception as e:
                        print(f'Error with {el["query"]}')
                        outputs.append(None)
                        continue
                else:
                    sql = None
                    sql_with_values = None
                instance = predictor._dataset_reader.text_to_instance(
                    utterance=el["question"], db_id=el["db_id"], sql=sql, sql_with_values=sql_with_values
                )
                # There is a bug that if we run with batch_size=1, the predictions are different.
                if i == 0:
                    instance_0 = instance
                if instance is not None:
                    predictor._dataset_reader.apply_token_indexers(instance)
                    with torch.cuda.amp.autocast(enabled=True):
                        out = predictor._model.forward_on_instances(
                            [instance, instance_0]
                        )
                        print(out[0].keys())
                        #print(out[0]['beam_scores'].size(), out[0]['beam_scores'].sum())
                        #pred = out[0]["sql_list"]
                        pred = 'Dummy prediction'

                else:
                    pred = "NO PREDICTION"
                if instance is not None:
                    g.write(f"{pred}\t{instance['db_id'].metadata}\n")
                else:
                    g.write(f"{pred}\t{el['db_id']}\n")
                outputs.append(out[0])
                #if i == 50:
                #    exit(0)
    with open('outputs.pickle', 'wb') as f:
        pickle.dump(outputs, f)
    assert len(outputs) == len(dev_json)

if __name__ == "__main__":
    main()
