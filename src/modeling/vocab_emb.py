'''
Author: wbx221240 221240001@smail.nju.edu.cn
Date: 2025-09-01 15:17:09
LastEditors: wbx221240 221240001@smail.nju.edu.cn
LastEditTime: 2025-09-07 23:43:49
FilePath: /llama3/src/modeling/vocab_emb.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.nn import Parameter

"""Parallel Vocab Embedding
- First, the Vocab table is divided into w[world_size] disjoint partitions, each with size of (n, e), n = v // w
- For each partition r, it contains a partial Vocab Embedding Table Tr_r, it only embed the input ids in the range of [r * n, (r + 1) * n - 1]. It leave other
place with 0.
- After that, different partitions can run parallelly and their results can be summed up to be the whole embedding of input of 
size (b, s, e)
"""

class ParallelVocabEmbedding(nn.Module):
    """Parallel Vocab Embedding module
    This is a simplified version of the practical one, \
        which shards the vocabulary embedding table into `world_size` partitions in a process group, and \
        each rank in that process group only handles one partition of it, thus \
        in pratical, we can apply the large vocabulary embedding in parallel and then reduce them together.
    However, for this simpler module, you only need to implement the jobs for any single rank, and \
        don't have to handle the reduce operation or any other parallelism stuff.
    """
    
    def __init__(self, 
        vocab_size: int, 
        emb_size: int,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize Parallel Vocab Embedding module
        
        Args:
            vocab_size(int): vocabulary size
            emb_size(int): embedding size
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean of the normal distribution
            init_std(float, default = 1.0): standard deviation of the normal distribution
            init_base_seed(int, default = 42): the base seed for the initialization (the real seed will be base seed + rank)
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        assert not vocab_size % world_size, "The vocab size is not divisible by world size!"
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.emb_size = emb_size
        self.n = self.vocab_size // self.world_size
        self.rank = rank
        factory_kwargs = {"device": device, "dtype": dtype}
        self.vocab_table = Parameter(torch.zeros((world_size, vocab_size // world_size, emb_size), **factory_kwargs))
        self.init_kwargs = {"init_mean": init_mean, "init_std": init_std, "init_base_seed": init_base_seed}
        self.reset_parameters()

        # raise NotImplementedError("TODO: Assignment1 - Task2")
        
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """The forward pass for Parallel Vocab Embedding module
        
        Args:
            input_ids(torch.LongTensor): input ids, with shape: (batch_size, seq_len)
        
        Returns:
            output(torch.Tensor): output embedding tensor, with shape: (batch_size, seq_len, emb_size)
        """
        batch_size, seq_len = input_ids.shape
        embedding = torch.zeros((batch_size, seq_len, self.emb_size), dtype=self.vocab_table.dtype, device=input_ids.device)
        emb_pos = torch.where((self.rank * self.n <= input_ids) & (input_ids < (self.rank + 1) * self.n))
        embedding[emb_pos] = self.vocab_table[self.rank][input_ids[emb_pos] % self.n]
        return embedding.to(self.vocab_table.dtype)

        # raise NotImplementedError("TODO: Assignment1 - Task2")
        
    def reset_parameters(self) -> None:
        """Initialize learnable embedding parameters for Vocab Embedding from a normal distribution"""
        init_base_seed = self.init_kwargs["init_base_seed"] + self.rank
        init_mean = self.init_kwargs["init_mean"]
        init_std = self.init_kwargs["init_std"]
        torch.manual_seed(init_base_seed)
        nn.init.normal_(self.vocab_table[self.rank], init_mean, init_std)
        # raise NotImplementedError("TODO: Assignment1 - Task2")
        