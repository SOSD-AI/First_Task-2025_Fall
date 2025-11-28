import regex
from typing import Dict, List, Tuple, Iterable, Iterator, Optional

# 必须与 BPEtemp3.py 中的正则保持一致
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        # id <-> bytes
        self.id_to_bytes: Dict[int, bytes] = dict(vocab)
        self.bytes_to_id: Dict[bytes, int] = {v: k for k, v in vocab.items()}

        # merges: (bytes, bytes) -> rank (priority, lower is better)
        self.merges = merges
        self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # special tokens
        self.special_tokens = special_tokens or []
        self.special_set = set(self.special_tokens)

        # 预分词 regex
        special_tokens_pattern = (
            "|".join([regex.escape(tok) for tok in self.special_tokens])
            if self.special_tokens
            else ""
        )
        if special_tokens_pattern:
            pattern = f"({special_tokens_pattern})|{GPT2_PAT}"
        else:
            pattern = GPT2_PAT
            
        self._pretokenizer = regex.compile(pattern, flags=regex.UNICODE)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        vocab: Dict[int, bytes] = {}
        # 读取 vocab.txt (格式: ID \t HexString)
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split("\t")
                if len(parts) == 2:
                    idx = int(parts[0])
                    vocab[idx] = bytes.fromhex(parts[1])

        merges: List[Tuple[bytes, bytes]] = []
        # 读取 merges.txt (格式: HexLeft HexRight)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((bytes.fromhex(parts[0]), bytes.fromhex(parts[1])))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _bpe_bytes_to_ids(self, token_bytes: bytes) -> List[int]:
        # 1. 初始: 每个字节作为一个 token
        # ids 是一个包含 bytes 对象的列表，后续会合并
        parts: List[bytes] = [bytes([b]) for b in token_bytes]
        
        while len(parts) >= 2:
            # 找到当前 parts 中所有相邻对中，rank 最小（优先级最高）的一对
            min_rank = float("inf")
            min_idx = -1
            
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                rank = self.merge_ranks.get(pair)
                if rank is not None:
                    if rank < min_rank:
                        min_rank = rank
                        min_idx = i
            
            # 如果没有可以合并的 pair，结束
            if min_idx == -1:
                break
                
            # 执行合并: parts[min_idx] + parts[min_idx+1] -> merged
            merged_token = parts[min_idx] + parts[min_idx+1]
            parts[min_idx] = merged_token
            del parts[min_idx+1] # 删除被合并的后一项

        # 将最终的 bytes 列表转换为 int IDs
        ids: List[int] = []
        for p in parts:
            if p in self.bytes_to_id:
                ids.append(self.bytes_to_id[p])
            else:
                # Fallback (理论上不应该发生，因为初始化包含了所有单字节)
                for b in p:
                    ids.append(self.bytes_to_id.get(bytes([b]), 0))
        return ids

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        # 使用与训练相同的正则进行 split
        for m in self._pretokenizer.finditer(text):
            token_str = m.group(0)
            if not token_str: continue

            # 如果是 special token，直接查找 ID
            if token_str in self.special_set:
                token_bytes = token_str.encode("utf-8")
                if token_bytes in self.bytes_to_id:
                    ids.append(self.bytes_to_id[token_bytes])
                continue

            # 普通文本：先 utf-8 编码，再跑 BPE
            token_bytes = token_str.encode("utf-8")
            ids.extend(self._bpe_bytes_to_ids(token_bytes))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for tid in self.encode(text):
                yield tid

    def decode(self, ids: List[int]) -> str:
        res_bytes = bytearray()
        for tid in ids:
            if tid in self.id_to_bytes:
                res_bytes.extend(self.id_to_bytes[tid])
        # 使用 'replace' 处理无效 utf-8 序列
        return res_bytes.decode("utf-8", errors="replace")