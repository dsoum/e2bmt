#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-src=./data/corpus.train.en --train-tgt=./data/corpus.train.bn --dev-src=./data/RisingNews.valid.en --dev-tgt=./data/RisingNews.valid.bn --vocab=vocab.json --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python3 run.py decode model.bin ./data/RisingNews.test.en ./data/RisingNews.test.bn outputs/test_outputs.txt
elif [ "$1" = "dev" ]; then
        CUDA_VISIBLE_DEVICES=0 python3 run.py decode model.bin ./data/RisingNews.valid.en ./data/RisingNews.valid.bn outputs/dev_outputs.txt 
elif [ "$1" = "train_local" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-src=./data/RisingNews.train.en --train-tgt=./data/RisingNews.train.bn --dev-src=./data/RisingNews.valid.en --dev-tgt=./data/RisingNews.valid.bn --vocab=vocab.json --lr=5e-4 --max-epoch=1 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
elif [ "$1" = "train_debug" ]; then
	python3 run.py train --train-src=./data/RisingNews.test.en --train-tgt=./data/RisingNews.test.en --dev-src=./data/RisingNews.valid.en --dev-tgt=./data/RisingNews.valid.bn --vocab=vocab.json --lr=5e-4 --max-epoch=1
elif [ "$1" = "test_local" ]; then
    python3 run.py decode model.bin ./data/test.zh ./data/test.en outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python3 vocab.py --train-src=./data/corpus.train.en --train-tgt=./data/corpus.train.bn vocab.json
elif [ "$1" = "vocab_debug" ]; then
	python3 vocab.py --train-src=./data/RisingNews.test.en --train-tgt=./data/RisingNews.test.bn vocab.json
else
	echo "Invalid Option Selected"
fi
