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
        temperature: float = 0.6,
        top_p: float = 0.9,
        logitbias: dict = None,
        k: int = 1,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompts (List[str]): List of prompt.
            max_gen_len (int): Maximum length of the generated new tokens.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logitbias (dict, optional): Tokens and the associated biases to be added to appropriate output logits.  Defaults to None.
            k (int): top-k logprobs and tokens to return. Defaults to 1.
        """
        inputs = self.tokenizer(prompts, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=max_gen_len, return_dict_in_generate=True,
                                  output_logits=True, pad_token_id=self.tokenizer.eos_token_id)
        # Assume batch_size is always 1
        output_token = out['sequences'].to('cpu')
        output_logit = out['logits'][0].to('cpu')

        # print(self.model)
        # print(output_token)
        # print(output_logit.size())
        # print(self.tokenizer.vocab_size)
        # print(len(self.tokenizer))
        return output_token, output_logit


if __name__ == '__main__':
    llm = LLMAPI()
    output_token, output_logit = llm.generate(["Hello, my name is"])
