import torch
from .utils import PhobertTokenizerFast
from transformers import AutoModelForQuestionAnswering

class phoBERT:
    def __init__(self, 
                 model_pretrained: str, 
                 tokenizer_pretrained: str,
                 device: str) -> None:

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_pretrained)
        self.tokenizer = PhobertTokenizerFast.from_pretrained(tokenizer_pretrained)
        self.device = device
        self.model = self.model.to(self.device)

    def answer(self, question: str, context: str) -> str:
        _question = [question]
        _context = [context]

        encodings = self.tokenizer(_context, _question, 
                              truncation=True, padding=True)

        with torch.no_grad():
            input_ids = torch.tensor(encodings['input_ids']).to(self.device)
            attention_mask = torch.tensor(encodings['attention_mask']).to(self.device)

            outputs = self.answermodel(input_ids, attention_mask=attention_mask)
            start_idx = torch.argmax(outputs['start_logits']).item()
            end_idx = torch.argmax(outputs['end_logits']).item()

            answer = context[start_idx: end_idx]

        return answer