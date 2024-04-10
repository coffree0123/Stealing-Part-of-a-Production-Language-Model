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


#######################################################################################################################
# Paper Section 6.1 Method
#######################################################################################################################
def binary_search_extraction(llm, prompts, guess_token=0, bias=100, error=0.01):
    """
    Method described in section 6.1 (Guess one token version)
    Incrementally construct logit vector by finding logit bias for each token that
    causes it to be the the top-token (output with probability of 1 when temperature is 0).
    """
    vocab_size = len(llm.tokenizer)

    # Assume we can only get the top token
    top_token, original_logits = llm.generate(
        prompts=prompts,
        max_gen_len=1,
        temperature=1e-5,
    )

    # We need top_logit so that we can calculate the remaining tokens' logit value.
    top_logit = original_logits[0][top_token]

    # Guess every token compare to the top token
    if guess_token == top_token:
        return top_logit

    alpha = -bias
    beta = bias
    while beta - alpha > error:
        mid = (alpha + beta) / 2

        logitbias = {guess_token:mid}
        cur_top_token, _ = llm.generate(
            prompts=prompts,
            max_gen_len=1,
            temperature=1e-5,
            logitbias=logitbias,
        )

        # I modify the code here to make it consistent with my logitbias implementation.
        if cur_top_token == top_token:
            alpha = mid
        else:
            beta = mid

    guess_logit = top_logit - ((beta + alpha) / 2)

    # Verify the guess
    print(f"Token: {guess_token}, Original: {original_logits[0][guess_token]}, Guess: {guess_logit}")
    assert abs(guess_logit - original_logits[0][guess_token]) <= error
    
    return guess_logit