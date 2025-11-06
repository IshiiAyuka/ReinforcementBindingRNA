# ZHMolGraph

RNA-protein interactions are critical to various life processes, including fundamental translation and gene regulation. Identifying these interactions is vital for understanding the mechanisms underlying life processes. Then, ZHMolGraph is an advanced pipeline that integrates graph neural network sampling strategy and unsupervised large language models to enhance binding predictions for novel RNAs and proteins.


# Setting up ZHMolGraph and Predicting RNA-protein interactions

## Requirements

- We provide the script and model for validating the results of ZHMolGraph. Any machines with a GPU and an Ubuntu system should work.

- We recommend using Anaconda to create a virtual environment for this project.

- you will need a major software package: `pytorch`. The following commands will create a virtual environment and install the necessary packages. Note that we install the GPU version of PyTorch (`torch==1.8.1+cu11`) for training purpose.

```bash
conda create -n ZHMolGraphPytorch-1.8 python=3.8
conda activate ZHMolGraphPytorch-1.8
pip install tqdm
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard
pip install jupyter
pip install rna-fm
```

- All Python modules and corresponding versions required for ZHMolGraph are listed here: requirements.txt

- Use pip install -r requirements.txt to install the related packages. 


# Code and Data

## Data Files
All data files are available here: https://zenodo.org/records/14511350

- data/interactions/: Contains RNA-protein interactions data.
- data/Mol2Vec/: Contains the embeddings from LLMs.
- trained_model/: Contains the trained model.

## Code 

Here we describe the Jupyter Notebooks scripts used in ZHMolGraph.
### Training and testing ZHMolRPGraph on the benchmark datasets in the form five-fold cross validation 

#### Dataset NPInter2

1_NPInter2_result_validation.ipynb: We execute a five-fold cross-validation on benchmark dataset NPInter2.

#### Dataset RPI7317

2_RPI7317_result_validation.ipynb: We execute a five-fold cross-validation on benchmark dataset RPI7317.


### Testing ZHMolRPGraph on the unknown nodes dataset TheNovel
3_TheNovel_unknown_dataset_validation.ipynb: We conduct testing on the unseen nodes dataset NPInter5 using the trained models generated from the five-fold cross-validation on the benchmark dataset NPInter2.

### Predict the binding probability between a pair of given RNA and protein sequence
python predict_RPI.py -r example/RNA_seq.fasta -p example/protein_seq.fasta -j test -o example/Result

# Contact

If you have any comments, questions or suggestions about the ZHMolRPGraph, please contact:

- Yunjie Zhao       E-mail: yjzhaowh@ccnu.edu.cn
- Haoquan Liu       E-mail: liuhaoquan@mails.ccnu.edu.cn
