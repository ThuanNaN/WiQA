import torch
from transformers import AlbertForQuestionAnswering, AlbertTokenizerFast

class Albert:
    def __init__(self, 
                 model_pretrained: str, 
                 tokenizer_pretrained: str,
                 device: str) -> None:

        self.model = AlbertForQuestionAnswering.from_pretrained(model_pretrained)
        self.tokenizer = AlbertTokenizerFast.from_pretrained(tokenizer_pretrained)
        self.device = device
        self.model = self.model.to(self.device)

    def answer(self, question: str, context: str) -> str:
        _question = [question]
        _context = [context]

        encodings = self.tokenizer(_context, _question, truncation=True, 
                                   padding='max_length', return_tensors='pt')
        
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            start_idx_token = torch.argmax(outputs['start_logits']).item()
            end_idx_token = torch.argmax(outputs['end_logits']).item()

            char_span_start = encodings.token_to_chars(start_idx_token)
            char_span_end = encodings.token_to_chars(end_idx_token)

            if char_span_start is None or char_span_end is None:
                answer = ""
            else:
                if char_span_start.start <  char_span_end.end:
                    answer = context[char_span_start.start: char_span_end.end]
                else:
                    answer = ""
                    
        return answer