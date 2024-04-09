from attack.llm_api import LLMAPI
from attack.llm_attack import load_prompts, get_q, h_dim_extraction, layer_extraction


if __name__ == '__main__':
    # Hyperparameters you can change.
    model_name = "EleutherAI/gpt-neo-125m"
    # This parameter need to be larger than the hidden dimension of the LLM model.
    num_samples = 1000

    # Start the attack
    llm = LLMAPI(model_name)
    prompts = load_prompts()

    q = get_q(
        llm=llm,
        prompts=prompts,
        n=num_samples,
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
