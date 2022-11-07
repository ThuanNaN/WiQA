import subprocess
import shlex
import yaml
from utils import execute, load_config


lucene_indexer_cmd = """
python -m pyserini.index.lucene --storeContents \
                                --bm25.accurate \
"""

cmd_args = shlex.split(lucene_indexer_cmd)
config = load_config(key="lucene")

for k, v in config.items():
    cmd_args += [k] + [v]

for path in execute(cmd_args):
    print(path, end="")