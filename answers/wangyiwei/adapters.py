from typing import Dict, List, Tuple
import os

# 导入你的模块
from BPEtemp3 import BPE 
from tokenizer import Tokenizer

def run_train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Problem 3 Adapter
    """
    # BPE 返回的是 (merges, vocab_dict)
    merges_list, int2utf = BPE(input_path, vocab_size, special_tokens)
    
    # 题目要求返回: vocab (dict[int, bytes]), merges (list[tuple[bytes, bytes]])
    return int2utf, merges_list

def get_tokenizer() -> Tokenizer:
    """
    Problem 5 Adapter
    """
    # 假设训练生成的文件保存在 data/ 目录下
    # 注意：运行此函数前，通常需要先运行过训练，或者 data/ 下已有预置文件
    vocab_path = os.path.join("data", "vocab.txt")
    merges_path = os.path.join("data", "merges.txt")
    special_tokens = ["<|endoftext|>"]
    
    # 如果文件不存在，这里可能会报错。
    # 在测试流程中，通常会先调用 train 生成文件，或者直接实例化 Tokenizer
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)