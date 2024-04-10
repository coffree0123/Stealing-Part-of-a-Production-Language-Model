## Stealing-part-of-an-LM
An unofficial implementation of ["Stealing Part of a Production Language Model"](https://arxiv.org/abs/2403.06634).
This implementation support different language models provided in transformers library.

### Requirements
Python 3.9 and run the following command:
```{bash}
pip install -r requirements.txt
```

### How to run the code
```{bash}
python main.py --model_name=EleutherAI/gpt-neo-125m --num_samples=1000 --guess_token=1
```

This repo is modified from [Github repo](https://github.com/sramshetty/stealing-part-of-an-LM).
