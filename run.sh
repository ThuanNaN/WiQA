#!/bin/sh

# while getopts s:e: flag
# do
#     case "${flag}" in
#         s) start_id=${OPTARG};;
#         e) end_id=${OPTARG};;
#     esac
# done

### Uncomment two lines below if you use conda enviroment
eval "$(conda shell.bash hook)"
conda activate zalo_qa_env

DATASET_FOLDER_PATH = "./dataset"
CORPUS_PICKLE_PATH="./dataset/corpus.pkl"

# check if folder dataset exists
if [ -e $DATASET_FOLDER_PATH ]
then
    echo "Already have dataset folder"
else
    echo "Create dataset folder"
    mkdir $DATASET_FOLDER_PATH
fi

cd $DATASET_FOLDER_PATH
# https://drive.google.com/file/d/1-wi2eY4XIg9v_COahiwDs6vt17zy2e08/view?usp=share_link
gdown --id 1-wi2eY4XIg9v_COahiwDs6vt17zy2e08
cd ..

# create corpus
if [ -e $CORPUS_PICKLE_PATH ]
then
    echo "Already have corpus.pkl"
else
    echo "Read corpus.pkl..."
    python utils/read_corpus.py
fi

# create submit file
python submit.py