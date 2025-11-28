import os
from collections import defaultdict
import multiprocessing
import regex
import heapq
import time
import psutil

# 统一的 GPT-2 Regex，确保训练和 Tokenizer 使用完全一致的正则
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *args, **kwargs): pass
        def update(self, n=1): pass
        def close(self): pass

# ========== 工具函数：按 <|endoftext|> 切分大文件 ==========

def find_chunk_boundaries(file, desire_num_chunks: int, split_special_token: bytes):
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desire_num_chunks <= 0 or file_size == 0:
        return [0, file_size]

    chunk_size = max(1, file_size // desire_num_chunks)
    chunk_boundaries = [i * chunk_size for i in range(desire_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        inital_position = chunk_boundaries[bi]
        file.seek(inital_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = inital_position + found_at
                break
            inital_position += len(mini_chunk)
    return sorted(set(chunk_boundaries))

# ========== 工具函数：按特殊 token 粗粒度切块 ==========

def chunk_text(chunk_block, special_tokens):
    if not special_tokens:
        yield chunk_block
        return

    pattern = b"(" + b"|".join([regex.escape(token.encode("utf-8")) for token in special_tokens]) + b")"
    regex_chunk = regex.compile(pattern)
    chunks = regex.split(regex_chunk, chunk_block)
    for chunk in chunks:
        if chunk:
            yield chunk

# ========== 单个分块的处理：预分词 + 建图 ==========

def process_chunk(start, end, special_tokens, input_path):
    pair_positions = defaultdict(list)
    token_dict = {}
    pair_counter = defaultdict(int)
    next_node = {}
    prev_node = {}

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_block = f.read(end - start)

        texts = chunk_text(chunk_block, special_tokens)

        # 使用统一的 GPT-2 正则
        special_tokens_pattern = "|".join([regex.escape(token) for token in special_tokens]) if special_tokens else ""
        if special_tokens_pattern:
            pattern = f"({special_tokens_pattern})|{GPT2_PAT}"
        else:
            pattern = GPT2_PAT
        
        compiled_pattern = regex.compile(pattern, flags=regex.UNICODE)

        token_idx = start 

        for text in texts:
            # text 是 bytes
            text_str_safe = text.decode("utf-8", errors='ignore')
            text_split = compiled_pattern.finditer(text_str_safe)
            
            for token in text_split:
                token_str = token.group(0)
                token_bytes = token_str.encode('utf-8')
                
                # 遇到特殊 token：跳过并推进位置
                if token_str in special_tokens:
                    token_idx += len(token_bytes)
                    continue

                prev_token = None
                for c in token_bytes:
                    token_dict[token_idx] = c
                    next_node[token_idx] = None
                    prev_node[token_idx] = None

                    if prev_token is not None:
                        pair_positions[(prev_token, c)].append(token_idx - 1)
                        pair_counter[(prev_token, c)] += 1
                        next_node[token_idx - 1] = token_idx
                        prev_node[token_idx] = token_idx - 1

                    token_idx += 1
                    prev_token = c

    return pair_positions, token_dict, pair_counter, next_node, prev_node

# ========== 初始化：并行预分词 + 全局统计 ==========

def BPE_init(input_path: str, vocab_size: int, special_tokens: list[str], vocab_tot):
    utf2int = {}
    int2utf = {}

    for i in range(256):
        utf2int[bytes([i])] = i
        int2utf[i] = bytes([i])

    for i in special_tokens:
        utf2int[i.encode("utf-8")] = vocab_tot
        int2utf[vocab_tot] = i.encode("utf-8")
        vocab_tot += 1

    with open(input_path, "rb") as f:
        num_processes = max(1, multiprocessing.cpu_count() // 2)
        # 只有当文件包含 <|endoftext|> 时才按此切分，否则整个文件一块处理
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    task = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            task.append((start, end, special_tokens, input_path))

    global_pair_positions = defaultdict(list)
    global_token_dict = {}
    global_pair_counter = defaultdict(int)
    global_next = {}
    global_prev = {}

    print(f"[INFO] 开始并行预分词，共 {len(task)} 个分块")

    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in pool.starmap(process_chunk, task):
            pair_position, token_dict, pair_counter, next_, prev_ = result

            for pair, positions in pair_position.items():
                if pair in global_pair_positions:
                    global_pair_positions[pair].extend(positions)
                else:
                    global_pair_positions[pair] = list(positions)

            global_token_dict.update(token_dict)
            for pair, count in pair_counter.items():
                global_pair_counter[pair] += count
            global_next.update(next_)
            global_prev.update(prev_)

    return (global_pair_positions, global_token_dict, global_pair_counter, 
            global_next, global_prev, int2utf, utf2int, vocab_tot)

# ========== 单次 BPE 合并 ==========

def BPE_merge(pair_positions, token_dict, pair_counter, next_node, prev_node, heap, vocab_tot, int2utf, utf2int):
    pair_chosen = None
    pair_freq = 0

    while heap:
        neg_cnt, pair = heapq.heappop(heap)
        cnt = -neg_cnt
        if cnt == 0: continue
        if pair_counter[pair] != cnt: continue # 懒删除

        # Tie-break logic
        same_pairs = [pair]
        freq = cnt
        while heap and -heap[0][0] == freq:
            ncnt, p2 = heapq.heappop(heap)
            if pair_counter[p2] != -ncnt: continue
            same_pairs.append(p2)

        def pair_name(p):
            return (int2utf[p[0]], int2utf[p[1]])

        pair_chosen = max(same_pairs, key=pair_name)
        pair_freq = freq

        for p in same_pairs:
            if p is pair_chosen: continue
            heapq.heappush(heap, (-pair_counter[p], p))
        break

    if pair_chosen is None or pair_freq == 0:
        return None

    pair0, pair1 = pair_chosen
    new_token_id = vocab_tot
    new_token_bytes = int2utf[pair0] + int2utf[pair1]
    utf2int[new_token_bytes] = new_token_id
    int2utf[new_token_id] = new_token_bytes

    prev_neighbors = set()
    next_neighbors = set()

    for idx0 in list(pair_positions[pair_chosen]):
        idx1 = next_node.get(idx0)
        if idx1 is None: continue
        if token_dict.get(idx0) != pair0 or token_dict.get(idx1) != pair1: continue

        prev_idx = prev_node.get(idx0)
        next_idx = next_node.get(idx1)

        prev_token = token_dict.get(prev_idx) if prev_idx is not None else None
        next_token = token_dict.get(next_idx) if next_idx is not None else None

        # 减少旧邻居计数
        if prev_idx is not None and prev_token not in (None, -1):
            if next_node.get(prev_idx) == idx0 and token_dict.get(idx0) == pair0:
                old_p = (prev_token, pair0)
                pair_counter[old_p] -= 1
                heapq.heappush(heap, (-pair_counter[old_p], old_p))

        if next_idx is not None and next_token not in (None, -1):
            if next_node.get(idx1) == next_idx and token_dict.get(idx1) == pair1:
                old_p = (pair1, next_token)
                pair_counter[old_p] -= 1
                heapq.heappush(heap, (-pair_counter[old_p], old_p))

        # 增加新邻居计数
        if prev_idx is not None and prev_token not in (None, -1):
            new_p = (prev_token, new_token_id)
            pair_counter[new_p] += 1
            pair_positions[new_p].append(prev_idx)
            prev_neighbors.add(prev_token)

        if next_idx is not None and next_token not in (None, -1):
            new_p = (new_token_id, next_token)
            pair_counter[new_p] += 1
            pair_positions[new_p].append(idx0)
            next_neighbors.add(next_token)

        # 更新链表
        if prev_idx is not None: next_node[prev_idx] = idx0
        prev_node[idx0] = prev_idx
        next_node[idx0] = next_idx
        if next_idx is not None: prev_node[next_idx] = idx0

        token_dict[idx0] = new_token_id
        token_dict[idx1] = -1

    pair_counter[pair_chosen] = 0
    pair_positions[pair_chosen] = []

    for prev_token_id in prev_neighbors:
        p = (prev_token_id, new_token_id)
        heapq.heappush(heap, (-pair_counter[p], p))
    for next_token_id in next_neighbors:
        p = (new_token_id, next_token_id)
        heapq.heappush(heap, (-pair_counter[p], p))

    return pair_chosen

# ========== 导出函数 ==========

def export2file(vocabulary, bytes_merge_list, out_dir="data"):
    vocab_path = os.path.join(out_dir, "vocab.txt")
    merges_path = os.path.join(out_dir, "merges.txt")
    os.makedirs(out_dir, exist_ok=True)

    with open(vocab_path, "w", encoding="utf-8") as f:
        for idx, b in vocabulary.items():
            f.write(f"{idx}\t{b.hex()}\n") # ID \t HexString

    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in bytes_merge_list:
            f.write(f"{a.hex()} {b.hex()}\n") # HexLeft HexRight

# ========== BPE 主入口 ==========

def BPE(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab_tot = 256
    result = BPE_init(input_path, vocab_size, special_tokens, vocab_tot)
    
    pair_positions, token_dict, pair_counter, next_node, prev_node, int2utf, utf2int, vocab_tot = result

    heap = []
    merge_list = []
    
    for pair, count in pair_counter.items():
        if count > 0:
            heapq.heappush(heap, (-count, pair))

    pbar = tqdm(total=vocab_size - vocab_tot)
    while vocab_tot < vocab_size:
        if not heap: break
        vocab_tot += 1
        pair_merged = BPE_merge(pair_positions, token_dict, pair_counter, next_node, prev_node,
                                heap, vocab_tot, int2utf, utf2int)
        if pair_merged is None: break
        merge_list.append(pair_merged)
        pbar.update(1)
    pbar.close()

    merge_list_bytes = []
    for pair in merge_list:
        merge_list_bytes.append((int2utf[pair[0]], int2utf[pair[1]]))

    return merge_list_bytes, int2utf

if __name__ == "__main__":
    # 用于直接运行测试
    input_path = "TinyStoriesV2-GPT4-valid.txt" # 确保文件存在
    if os.path.exists(input_path):
        vocab_size = 5000
        special_tokens = ["<|endoftext|>"]
        print(f"Training BPE on {input_path}...")
        merge_list, vocabulary = BPE(input_path, vocab_size, special_tokens)
        export2file(vocabulary, merge_list)
        print("Done.")
    else:
        print(f"File {input_path} not found.")