import torch
import argparse
import time
import json
from model import Noob

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_vocab(model_path):
    """加载词汇表"""
    with open(f"{model_path}/vocab.json", 'r', encoding='utf-8') as f:
        stoi = json.load(f)
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode

def load_model(model_path):
    """加载预训练模型"""
    print(f"Loading model from {model_path}...")
    start_time = time.time()
    
    # 加载模型和词汇表
    model = Noob.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    encode, decode = load_vocab(model_path)
    
    elapsed_time = time.time() - start_time
    print(f"Model loaded in {elapsed_time:.2f} seconds")
    return model, encode, decode

def generate_text(model, encode, decode, prompt="", max_new_tokens=500, temperature=1.0):
    """生成文本
    Args:
        model: Noob模型实例
        encode: 编码函数
        decode: 解码函数
        prompt: 起始提示文本
        max_new_tokens: 最大生成token数
        temperature: 采样温度，控制生成文本的随机性
    """
    print(f"\nGenerating text with prompt: '{prompt}'")
    start_time = time.time()
    
    if prompt:
        # 将提示文本转换为tensor
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    output_text = decode(output_ids)
    
    elapsed_time = time.time() - start_time
    print(f"\nText generated in {elapsed_time:.2f} seconds")
    return output_text

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained Noob model')
    parser.add_argument('--model_path', type=str, default='noob_model',
                       help='Path to the saved model directory')
    parser.add_argument('--prompt', type=str, default="",
                       help='Starting prompt for text generation')
    parser.add_argument('--max_tokens', type=int, default=500,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (0.0 to 2.0)')
    args = parser.parse_args()

    try:
        # 加载模型和词汇表函数
        model, encode, decode = load_model(args.model_path)
        
        # 生成文本
        generated_text = generate_text(
            model,
            encode,
            decode,
            args.prompt,
            args.max_tokens,
            args.temperature
        )
        
        # 输出生成的文本
        print("\nGenerated Text:")
        print("=" * 50)
        print(generated_text)
        print("=" * 50)
        
    except FileNotFoundError:
        print(f"Error: Model directory '{args.model_path}' not found.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()