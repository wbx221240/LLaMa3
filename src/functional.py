'''
Author: wbx221240 221240001@smail.nju.edu.cn
Date: 2025-09-01 15:17:09
LastEditors: wbx221240 221240001@smail.nju.edu.cn
LastEditTime: 2025-09-14 23:45:49
FilePath: /llama3/src/functional.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def matmul_with_importance(
    input: torch.Tensor,
    weight: torch.Tensor,
    probs: torch.Tensor,
    grad_output: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """matmul input and weight and return output (with optional grad_input, grad_weight whenever grad_output is given) 
    where only the important elements of the input tensor can be computed and gathered to the output tensor
    decided by the importance probability tensor, tuned by top_p and top_k
    
    Args:
        input (torch.Tensor): input tensor in the range of [-1, 1], with shape: [batch_size, seq_len, hidden_size]
        weight (torch.Tensor): weight tensor in the range of [-1, 1], with shape: [hidden_size, embed_size]
        probs (torch.Tensor): probability tensor in the range of [0, 1], with shape: [batch_size, seq_len]
        grad_output (Optional[torch.Tensor], optional): gradient for the output tensor, with shape: [t, hidden_size]. Defaults to None.
        num_heads (int): number of heads to split hidden_size
        top_p (float, [0., 1.]): only the elements with the probability equal or higher than top_p are important ones
        top_k (int, [1, ..., seq_len], optional): only the elements with the top_k highest probability are important ones
    
    Returns:
        output (torch.Tensor): output tensor, with shape: [t, num_heads, embed_size]
        grad_input (torch.Tensor, optional): gradient for the input tensor if grad_output is given, otherwise None
        grad_weight (torch.Tensor, optional): gradient for the weight tensor if grad_output is given, otherwise None
    """
    raise NotImplementedError("TODO: Assignment0 - Task1")


def apply_rotary_pos_emb(
    input: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor.
    
    Args:
        input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        cos(torch.Tensor): cos basis tensor, with shape: [seq_len, head_dim]
        sin(torch.Tensor): sin basis tensor, with shape: [seq_len, head_dim]
    
    Returns:
        output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = input.shape

    input = input.permute(0, 2, 1, 3)
    # print("input(before):", input)
    # input_even = input[..., ::2]
    # input_odd = input[..., 1::2]
    # input = torch.cat((input_even, input_odd), dim=-1)
    # print("input(after):", input)
    # permute the input to [batch_size, num_heads, seq_len, head_dim]
    position_ids = torch.arange(0, seq_len).repeat(batch_size, 1)
    cos = cos[position_ids].unsqueeze(1) # [seq_len, 1, head_dim] 
    sin = sin[position_ids].unsqueeze(1) 
    emb = input * cos + rotate_half(input) * sin
    # print(restore_interleave(emb.permute(0, 2, 1, 3)))
    return emb.permute(0, 2, 1, 3) # restore_interleave(emb.permute(0, 2, 1, 3))
    # raise NotImplementedError("TODO: Assignment1 - Task3")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the channels of the input tensoray, this is a trick of LlaMA. It
    splits the channels into two halves where the first half is even indexed and the second is odd. 
    They use rotate the second half to the first and reverse the sign

    Args:
        x (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1) 

def restore_interleave(x:torch.Tensor):
    d = x.shape[-1] // 2
    even, odd = x[..., :d], x[..., d:]
    return torch.stack([even, odd], dim=-1).reshape(*x.shape)