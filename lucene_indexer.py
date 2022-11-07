import subprocess
import shlex
import yaml
from utils import CONFIG, execute


if __name__ == '__main__':    
    lucene_indexer_cmd = """
    python -m pyserini.index.lucene --storeContents \
                                    --bm25.accurate \
    """

    cmd_args = shlex.split(lucene_indexer_cmd)
    config = CONFIG['lucene']

    for k, v in config.items():
        cmd_args += [f"--{k}"] + [str(v)]

    for path in execute(cmd_args):
        print(path, end="")