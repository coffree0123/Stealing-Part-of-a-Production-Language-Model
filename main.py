import argparse
from attack.llm_api import LLMAPI
from attack.llm_attack import load_prompts, get_q, h_dim_extraction, layer_extraction, binary_search_extraction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment config')
    parser.add_argument('--model_name', type=str, default="EleutherAI/gpt-neo-125m",
                        help='LLM name that you want to attack.')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate, this parameter need to be larger than LLM hidden dimension size.')
    parser.add_argument('--guess_token', type=int, default=1,
                        help='Token to guess in ch 6.1.')
    parser.add_argument('--bias', type=int, default=100,
                        help='Bias value in ch 6.1.')
    parser.add_argument('--error', type=float, default=0.01,
                        help='Error value in ch 6.1.')
    args = parser.parse_args()

    # Start the attack.
    llm = LLMAPI(args.model_name)
    prompts = load_prompts()

    q = get_q(
        llm=llm,
        prompts=prompts,
        n=args.num_samples,
        batch_size=1,
    )

    # Target hidden dimension: 768, for EleutherAI/gpt-neo-125m model
    u, s, s_dim, pred_dim = h_dim_extraction(
        q=q,
    )

    print("###### Ch 4.1 ###########")
    print(f"Hidden Dim: {pred_dim}")
    print("#######################")

    # Predicted weight matrix
    pred_w = layer_extraction(
        u=u,
        s=s,
        h_dim=pred_dim
    )

    print("###### Ch 4.2 ###########")
    print("W_tilde: ")
    print(pred_w)
    print("W_tilde size: ", pred_w.size())
    print("#######################")

    print("###### Ch 6.1 ###########")
    # Given one random prompt, guess the logits.
    guess_logit = binary_search_extraction(
        llm, ["Hello, my name is"], guess_token=args.guess_token, bias=args.bias, error=args.error)
    print("#######################")
