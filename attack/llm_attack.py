import torch

from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler


#######################################################################################################################
# Paper Section 4.1 Method
#######################################################################################################################
def get_q(
    llm,
    prompts: list[str],
    n: int = 5000,
    batch_size: int = 1,
):
    """
    Args:
        llm: LLM API object
        prompts: list of prompts
        text_key: key in dataset that corresponds to the prompts
        n (optional): the number of samples to produce logits for
        batch_size (optional): batch size for model inference
        logit_fn: method to use to compute logit vectors

    Returns:
        q: matrix of logit vectors with size (n, l)
    """
    assert batch_size == 1, "Currently, only works with batch size of 1"

    # Prepare random sample n prompts from dataset.
    random_sampler = RandomSampler(prompts, num_samples=n)
    dataloader = DataLoader(
        prompts,
        batch_size=batch_size,
        sampler=random_sampler,
    )

    return direct_logits(
        llm,
        n,
        dataloader,
        batch_size
    )


def direct_logits(
    llm,
    n,
    dataloader,
    batch_size: int = 1,
):
    """
    Args:
        llm: LLM API object
        n: number of prompts
        dataloader: dataloader that contains prompts as samples
        batch_size (optional): batch size for model inference

    Returns:
        q: matrix of logit vectors with size (n, l)
    """
    q = torch.zeros(n, len(llm.tokenizer))
    for i, prompts in enumerate(tqdm(dataloader)):
        _, logits = llm.generate(prompts)
        q[i * batch_size:i * (batch_size + 1)] = logits

    return q


def h_dim_extraction(
    q,
):
    """
    Args:
        q: matrix of logit vectors with size (n, l)

    Returns:
        u: unitary matrix
        s: singular values
        s_dim: log of the absolute singular values
        count: predicted hidden dimension size
    """
    # compute singular values and prepare them to find the multiplicative gap
    u, s, _ = torch.linalg.svd(q.T.to(torch.float64), full_matrices=False)
    s_dim = torch.log(s.abs())

    # avoid large drops in negative singular values from causing a larger h_dim to be predicted
    # do so by multiplying by the sign of the first number -> multiplicative gap remains negative
    # also the last singular value is 0 so avoid using it for argmax computation
    count = torch.argmax(
        torch.where(s_dim[:-2] >= 0, 1, -1) * (s_dim[:-2] - s_dim[1:-1])
    ).item() + 1

    return u, s, s_dim, count


#######################################################################################################################
# Paper Section 4.2 Method
#######################################################################################################################
def layer_extraction(u, s, h_dim):
    """
    Args:
        u: unitary matrix computed from `h_dim_extraction`
        s: singular values computed from `h_dim_extraction`
        h_dim: model's predict hidden dimension

    Returns:
        pred_w: predicted w
    """
    u, s = u[:, :h_dim].to("cuda"), s[:h_dim].to("cuda")
    pred_w = u @ torch.diag(s)

    return pred_w


def load_prompts():
    dataset = load_dataset("alespalla/chatbot_instruction_prompts")
    prompts = []
    for data in dataset['train']:
        # Escape empty prompts
        if (len(data['prompt']) == 0):
            continue
        prompts.append(data['prompt'])

    return prompts
