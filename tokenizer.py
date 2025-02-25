from transformers import PreTrainedTokenizer
from typing import List, Optional
import os
import json
import argparse

class NoobTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file=None, **kwargs):
        if vocab_file is None:
            # 默认词汇表，仅用于初始化
            with open('input.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
        else:
            # 从文件加载词汇表
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.stoi = json.load(f)
            self.itos = {int(i): ch for ch, i in self.stoi.items()}
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def get_vocab(self):
        return dict(self.stoi)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.stoi.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        return self.itos[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        vocab_file = os.path.join(save_directory, 'vocab.json')
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.stoi, f, ensure_ascii=False)
            
        return (vocab_file,)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train tokenizer')
    parser.add_argument('--save_dir', type=str, default='noob_model',
                        help='Directory to save the tokenizer')
    args = parser.parse_args()
    tokenizer = NoobTokenizer()
    tokenizer.save_pretrained(args.save_dir)
    print(f"Model saved to {args.save_dir}")