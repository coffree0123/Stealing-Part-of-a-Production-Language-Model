import torch
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMAPI():
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125m"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int = 1,
        temperature: float = 1e-5,
        logitbias: dict = None,
    ) -> Tuple[int, Optional[List[List[float]]]]:
        """
        Generate top_token and output logits for the given prompt.

        Args:
            prompts (List[str]): List of prompt.
            max_gen_len (int): Maximum length of the generated new tokens.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logitbias (dict, optional): Tokens and the associated biases to be added to appropriate output logits.  Defaults to None.
        """
        inputs = self.tokenizer(prompts, return_tensors="pt").to("cuda")

        out = self.model.generate(**inputs, max_new_tokens=max_gen_len, return_dict_in_generate=True,
                                  output_logits=True, pad_token_id=self.tokenizer.eos_token_id, temperature=temperature, do_sample=True)
        # Assume batch_size is always 1
        output_token = out['sequences'].to('cpu')
        output_logit = out['logits'][0].to('cpu')

        if (logitbias is not None):
            # Since transformers doesn't support logit bias, we need to manually implement it.
            for key, value in logitbias.items():
                output_logit[0][key] += value

        top_token = int(torch.argmax(output_logit, dim=-1)[0])
        # print(inputs)
        # print(output_token)
        # print(torch.argmax(output_logit))
        # print(self.model)
        # print(output_token)
        # print(output_logit.size())
        # print(self.tokenizer.vocab_size)
        # print(len(self.tokenizer))
        return top_token, output_logit


if __name__ == '__main__':
    llm = LLMAPI()
    output_token, output_logit = llm.generate(["Hello, my name is"])
