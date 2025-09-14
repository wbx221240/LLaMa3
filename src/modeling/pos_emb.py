import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..functional import apply_rotary_pos_emb


"""
The standard RoPE should return a pre-defined cosine and sine base vector, this vector matrix is 
(seq_len, hidden_size), after that, use apply_rotary_pos_emb function to apply the base vector to 
the input tensor: (batch_size, seq_len, hidden_size) -> (batch_size, nums_head, seq_len, head_size)
 
"""

class NTKAwareRoPE(nn.Module):
    """NTK-aware RoPE module
    This is a series variants of the RoPE modules based on NTK theory to enhance its extrapolation ability.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int,
        base: int = 10000,
        ratio: int = 1,
        dynamic: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
    ) -> None:
        """Initialize NTK-aware RoPE Module
        
        Args:
            dim (int): The dimension of the RoPE
            max_seq_len (int): The maximum sequence length used in training
            base (int, optional): The base of the NTK. Defaults to 10000.
            ratio (int, optional): The ratio of the NTK. Defaults to 1.
            dynamic (bool, optional): Whether to use dynamic mode. Defaults to False.
            dtype (torch.dtype, optional): The dtype of the RoPE. Defaults to torch.float32.
            device (str, optional): The device of the RoPE. Defaults to 'cpu'.
        """
        super().__init__()
        self.base = base
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.ratio = ratio
        self.extend_seq_len = int(max_seq_len * ratio)
        self.dynamic = dynamic
        self.factory_kwargs = {"device":device, "dtype": dtype}
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).to(dtype).to(device) / self.dim))
        
        self.register_buffer("inv_freq", inv_freq, persistent=False) # this should be set to False as it will change 
                                                                     # each time a longer sequence comes

        self._set_cos_sin_cache(seq_len=self.extend_seq_len, device=inv_freq.device, dtype=inv_freq.dtype, dynamic=dynamic, init=True)


        # raise NotImplementedError("TODO: Assignment1 - Task3")
    
    def _set_ratio(self, seq_len)->int:
        """find the lowest even integer that satisfies:
           es_ = ms * k_ >= s_ 

        Args:
            seq_len (int): current seq_len
        """
        ratio = math.ceil(seq_len / self.max_seq_len) 
        ratio += int(ratio % 2)
        return ratio
    
    def _set_cos_sin_cache(self, seq_len, device, dtype, dynamic, init=False):
        self.max_seq_len_cached = seq_len # this must be the extended size namely es
        # print(self.inv_freq, self.max_seq_len_cached, self.max_seq_len)      
        if seq_len > self.max_seq_len:
            scaling_factor = seq_len / self.max_seq_len
            base = self.base * (scaling_factor ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).to(self.factory_kwargs["device"]).to(self.factory_kwargs["dtype"]) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(0, self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        freq = torch.outer(t, self.inv_freq)
        
        emb = torch.cat((freq, freq), dim=-1)
        # print(emb)
        # the initial cos and sin cache must have the size of [es, hd] regardless of dynamic
        if dynamic or init:
            self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
            return self.cos_cached, self.sin_cached
        else:
            return emb.cos().to(dtype), emb.sin().to(dtype)
        

    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """The forward pass of the NTK-aware RoPE module
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
            offset(int, optional): The offset of the starting position index of the input tensor. Defaults to 0.
        
        Returns:
            output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        """

        batch_size, seq_len, num_heads, head_dim = input.shape
        device = input.device
        dtype = input.dtype
        cos, sin = None, None
        
        if seq_len + offset > self.extend_seq_len: 
            ratio = self._set_ratio(seq_len + offset)
            extend_seq_len = int(self.max_seq_len * ratio) # need a longer and new cos, sin vector  
            if self.dynamic:
                self.ratio = ratio
                self.extend_seq_len = extend_seq_len
            cos, sin = self._set_cos_sin_cache(extend_seq_len, device=self.factory_kwargs["device"], dtype=self.factory_kwargs["dtype"], dynamic=self.dynamic)
            cos, sin = cos[offset: seq_len + offset], sin[offset: seq_len + offset]
        else:
            cos, sin = self.cos_cached[offset: seq_len + offset], self.sin_cached[offset: offset + seq_len]
        # print(cos.dtype, cos)
        embeded = apply_rotary_pos_emb(input, cos, sin)
        return embeded.to(dtype)




        # raise NotImplementedError("TODO: Assignment1 - Task3")