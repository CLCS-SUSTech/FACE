from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F


def get_device_map(device: int):
    if device >= torch.cuda.device_count():
        return "auto"
    return device

class Model(object):
    def __init__(self, model_dir, device=0):
        self.device = get_device_map(device)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.model.requires_grad_(False)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )

    def generate(self, inputs, configs=None):
        inputs = self.tokenizer(
            inputs, return_tensors="pt", return_token_type_ids=False
        )
        for k, v in inputs.items():
            inputs[k] = v.cuda(self.device)
        outputs = self.model.generate(**inputs, **configs)
        output_texts = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_texts
        
    def forward(self, input_text: str, max_len=1024, return_tokens=False) -> torch.Tensor:
        token_ids = self.tokenizer(input_text, return_tensors='pt')['input_ids'] # shape: [1, L]
        max_len = min(token_ids.shape[1], max_len)
        inputs = token_ids[:, :max_len].cuda(self.device)
        logits = self.model(inputs)['logits'][0].cpu().to(torch.float32)
        probs = F.softmax(logits, dim=-1) # shape: [L, V]
        nlls = torch.zeros(probs.shape[0]-1, dtype=torch.float32) # NLLs for position i from 1 to L
        for i in range(probs.shape[0] - 1): # probs[i] is the probability distribution of the (i+1)th token, i.e., P(token[i+1] | history)
            nlls[i] = -torch.log(probs[i, token_ids[0, i+1]]) # NLL[i] = -log P(token[i+1] | history)
        if return_tokens:
            return logits, nlls, token_ids
        return logits, nlls
    
    def forward_batch(self, input_text: list[str], max_len=1024) -> torch.Tensor:
        token_ids = self.tokenizer(input_text, return_tensors='pt')['input_ids'] # shape: [B, L]
        max_len = min(token_ids.shape[1], max_len)
        inputs = token_ids[:, :max_len].cuda(self.device)
        logits = self.model(inputs)['logits'].cpu().to(torch.float32)
        #todo: implement batch nll calculation