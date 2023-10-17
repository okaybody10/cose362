# cose362

## TODO List
- [x] Preprocess
  - [x] DataLoader
  - [x] Re-labeling
  - [x] Handling json
- [ ] Compare 1, Vanilla
  - [ ] SVM
  - [ ] CRF
- [ ] Compare 2, BiLSTM
  - [ ] SVM
  - [ ] CRF
  - [ ] Accelerator
- [ ] Compare 3, Bert
  - [ ] SVM
  - [ ] CRF
  - [ ] Accelerator

# Preprocess
## DataLoader
* Use Dataloader in `pytorch.util.data`
* Our dataset is not formal, so we need to customize the dataloader to fit our dataset form.
* Maybe implement in `dataloader.py`

## Word Embedding
* We use bert (uncontextual) word embedding, so we have to get rid of positional embedding and sentence embedding
* If you use `model.embedding.word_embedding`, we can get only word embedding.
* Maybe implement in `preprocess.py`

## Re-labeling
* Done
  * Maybe need to debug?
* Currently this function is implemented in an ipynb file, but we will move it to a `preprocess.py` file.

# Vanilla
## SVM
* Linear Model
* If can, use several svm with kernel trick
* [Use this loss when implement the model](https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelMarginLoss.html)

## CRF
* Use pytorch-crf library to use it [Documents](https://pytorch-crf.readthedocs.io/en/stable/)
* [Tutorial](https://tutorials.pytorch.kr/beginner/nlp/advanced_tutorial.html)
* Maybe implement in vanilla.py

# LSTM
* [Tutorial](https://tutorials.pytorch.kr/beginner/nlp/advanced_tutorial.html)
* We don't use huggingface so we will use Pytorch Lightening
* Refer this [link](https://www.kaggle.com/code/megner/pytorch-lightning-lstm)

# KoBERT
* Use monologg/kobert, refer this [github](https://github.com/monologg/KoBERT-Transformers)
* We will use [Accelerator](https://huggingface.co/docs/accelerate/package_reference/accelerator) in huggingface
  * [Tutorial1](https://huggingface.co/docs/transformers/perf_train_gpu_one)
  * [Tutorial2](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one)

# Setting up Development Environment
* Recommend Anaconda to make virtual environment
1. Install Anaconda
1. Make Anaconda Environment & Activate
```shell
anaconda install python=3.7 pytorch=1.10.2 pandas scikit-learn

# to install Kobert
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master kobert-transformers
```
3. Create `./dataset` directory and place `NLNE2202211219.json` file from 국립국어원 말뭉치 in it.