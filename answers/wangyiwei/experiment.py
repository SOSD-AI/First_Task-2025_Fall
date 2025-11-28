import time
import os
from tokenizer import Tokenizer

def load_tokenizer():
    """加载之前训练好的 Tokenizer"""
    vocab_path = os.path.join("data", "vocab.txt")
    merges_path = os.path.join("data", "merges.txt")
    special_tokens = ["<|endoftext|>"]
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)

def get_sample_docs(filepath, num_docs=10):
    """从文件中读取前 num_docs 个文档"""
    docs = []
    # 也就是按照 <|endoftext|> 切分
    delimiter = "<|endoftext|>"
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        # 这里为了简单，读取整个文件内容然后 split。
        # 如果文件非常大（几个GB），建议用流式读取，但验证集通常不大。
        content = f.read()
        raw_docs = content.split(delimiter)
        
        # 过滤掉空字符串，取前 num_docs 个
        for d in raw_docs:
            if d.strip():
                docs.append(d.strip())
            if len(docs) >= num_docs:
                break
    return docs

def run_experiment():
    input_path = "TinyStoriesV2-GPT4-valid.txt"
    
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    print("正在加载 Tokenizer...")
    tokenizer = load_tokenizer()
    
    print(f"正在从 {input_path} 读取样本...")
    docs = get_sample_docs(input_path, 10)
    print(f"成功读取 {len(docs)} 个文档。")

    total_bytes = 0
    total_tokens = 0
    start_time = time.time()

    print("-" * 40)
    print(f"{'文档ID':<10} {'字节数':<10} {'Token数':<10} {'压缩率':<10}")
    print("-" * 40)

    for i, doc in enumerate(docs):
        # 1. 计算原始字节数
        doc_bytes = len(doc.encode("utf-8"))
        
        # 2. 编码并统计 Token 数量
        ids = tokenizer.encode(doc)
        num_tokens = len(ids)
        
        # 计算单个文档的压缩率
        ratio = doc_bytes / num_tokens if num_tokens > 0 else 0
        
        print(f"{i+1:<10} {doc_bytes:<10} {num_tokens:<10} {ratio:.2f}")

        total_bytes += doc_bytes
        total_tokens += num_tokens

    end_time = time.time()
    total_time = end_time - start_time

    # --- 统计结果 ---
    print("-" * 40)
    
    # (a) 计算整体压缩率 (bytes / token)
    avg_compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    print(f"\n[Problem 6a] 平均压缩率: {avg_compression_ratio:.2f} bytes/token")
    print(f"  (意味着平均每 {avg_compression_ratio:.2f} 个字节被压缩成 1 个 token)")

    # (b) 估算吞吐量 (bytes / second)
    throughput = total_bytes / total_time if total_time > 0 else 0
    print(f"[Problem 6b] 分词吞吐量: {throughput:.2f} bytes/second")
    print(f"  (总处理时间: {total_time:.4f} 秒)")

if __name__ == "__main__":
    run_experiment()