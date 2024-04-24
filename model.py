from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F


def get_device_map(device: int) -> str or int:
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
        
    def forward(self, prompt, content, content_token_length=1024) -> torch.Tensor:
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids']
        content_tokens = self.tokenizer(content, return_tensors='pt')['input_ids']
        prompt_tokens_length = prompt_tokens.shape[1]
        content_token_length = min(content_tokens.shape[1], content_token_length)
        inputs = torch.cat(
            [
                prompt_tokens, 
                content_tokens[:, 1: content_token_length]], # 1: to skip the first token [SEP]?
            dim=-1
        ).cuda(self.device)
        outputs = self.model(inputs)['logits'][0].cpu().to(torch.float32)
        logits = F.softmax(outputs, dim=-1)
        logits = logits[prompt_tokens_length - 1: content_token_length + prompt_tokens_length - 1]
        result = torch.zeros(logits.shape[0], dtype=torch.float32)
        for i in range(logits.shape[0]):
            result[i] = logits[i, content_tokens[0, i]]
        return result.numpy()
    

class ModelNoPrompt(Model):
    def __init__(self, model_dir, device=0):
        super().__init__(model_dir, device)

    def forward(self, content: str, content_token_length=1024) -> torch.Tensor:
        content_tokens = self.tokenizer(content, return_tensors='pt')['input_ids'] # shape: [1, L]
        content_token_length = min(content_tokens.shape[1], content_token_length)
        inputs = content_tokens[:, :content_token_length].cuda(self.device)
        logits = self.model(inputs)['logits'][0].cpu().to(torch.float32)
        probs = F.softmax(logits, dim=-1) # shape: [L, V]
        result = torch.zeros(probs.shape[0], dtype=torch.float32)
        for i in range(probs.shape[0]):
            result[i] = probs[i, content_tokens[0, i]]
        return result