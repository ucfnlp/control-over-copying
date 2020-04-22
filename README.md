# Controlling the Amount of Verbatim Copying in Abstractive Summarization

We provide the source code for the paper **"[Controlling the Amount of Verbatim Copying in Abstractive Summarization](https://arxiv.org/pdf/1911.10390.pdf)"**, accepted at AAAI'20. If you find the code useful, please cite the following paper. 

    @inproceedings{control-over-copying:2020,
     Author = {Kaiqiang Song and Bingqing Wang and Zhe Feng and Liu Ren and Fei Liu},
     Title = {Controlling the Amount of Verbatim Copying in Abstractive Summarization},
     Booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
     Year = {2020}}

## Goal

* Our system seeks to re-write a lengthy sentence, often the 1st sentence of a news article, to a concise, title-like summary. The average input and output lengths are 31 words and 8 words, respectively. 

* The code takes as input a text file with one sentence per line. It generates a text file ("summary.txt") in the working folder as the outputs, where each source sentence is replaced by a title-like summary.

* Example input and output are shown below. 
  > Belgian authorities are investigating the killing of two policewomen and a passerby in the eastern city of Liege on Tuesday as a terror attack, the country's prosecutor said.

  > Belgium probes killing of two policewomen as terror attack . 


## Dependencies

The code is written in Python (v3.7) and Pytorch (v1.3). We suggest the following environment:

* A Linux machine (Ubuntu) with GPU
* [Python (v3.7)](https://www.anaconda.com/download/)
* [Pytorch (v1.3)](https://pytorch.org/)
* [Pyrouge](https://pypi.org/project/pyrouge/)
* [pytorch-pretrained-bert](https://github.com/huggingface/transformers)

HINT: Notice that [pytorch-pretrained-bert](https://github.com/huggingface/transformers) may change their name and content during time. It is currently named as transformers.

To install [Python (v3.7)](https://www.anaconda.com/download/), run the command:
```
$ wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
$ bash Anaconda3-2019.10-Linux-x86_64.sh
$ source ~/.bashrc
```

To install [PyTorch (v1.3)](https://pytorch.org/) and its dependencies, run the below command.
```
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

To install [pytorch-pretrained-bert](https://github.com/huggingface/transformers) and its dependencies, run the below command.
```
$ pip install spacy ftfy==4.4.3
$ python -m spacy download en
$ pip install pytorch-pretrained-bert
``` 

To install [Pyrouge](https://pypi.org/project/pyrouge/), run the command below. Pyrouge is a Python wrapper for the ROUGE toolkit, an automatic metric used for summary evaluation.  
```
$ pip install pyrouge
```

## I Want to Generate Summaries..

1. Clone this repo. Download this ZIP  file ([`others.zip`](http://i2u.world/kqsong/model/aaai2020_kaiqiang_2/others.zip)) containing trained model. Move the ZIP file to the working folder and uncompress.
    ```
    $ git clone git@github.com:KaiQiangSong/control-over-copying.git
    $ mv others.zip control-over-copying
    $ cd control-over-copying
    $ unzip others.zip
    $ rm others.zip
    $ mkdir log
    ```

2. Generating Summaries with our joint parsing and generating summarization model trained on selected dataset including: gigaword (default), newsroom, cnndm (for CNN/DM-R), websplit.
    ```
    $ python run.py --do_test --inputFile data/test.txt
    ```
    Or if you want runing models other than that trained on gigaword:
    ```
    $ python run.py --do_test --dataset newsroom --inputFile data/test.txt
    ```
   
## I Want to Train the Model..
1. Training the Model with train files and validation files.
    ```
    $ python run.py --do_train --train_prefix data/train --valid_prefix data/valid
    ```

2. (Optional) Modify the training options.
    
    You might want to change the parameters used for training. These are specified in `./setttings/training/gigaword_8.json` and explained blow.
    
```
{
	"stopConditions":
	{
		"max_epoch":12,
		"earlyStopping":false,
		"rateReduce_bound":200000
	},
	"checkingPoints":
	{
		"checkMin":0,
		"checkFreq":2000,
		"everyEpoch":true
	}
}
```

HINT*: 200K batches (used for `rateReduce_bound`) with batch size of `8`, is slightly less than half of an epoch.
