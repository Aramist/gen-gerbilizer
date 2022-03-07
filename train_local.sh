#/bin/zsh

if [ -z $1 ]
then
    echo "Job id should be provided as a command line argument"
    exit 1
fi

CFG_PATH=$2
if [ -z $2 ]
then
    echo "No config file provided, using local.json..."
    CFG_PATH=configs/local.json
fi

source ~/.zshrc
conda activate vox
python3 generative/train.py \
    $1 \
    --config_path $CFG_PATH
