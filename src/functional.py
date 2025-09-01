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
    batch_size, seq_len, hidden_size = input.shape
    embed_size = weight.shape[-1]
    
    ## first, multi-head split for input A1 and weight W1
    head_size = hidden_size // num_heads
    input = input.view(batch_size, seq_len, num_heads, head_size)
    weight = weight.view(num_heads, head_size, embed_size)

    ## second, probability/importance mask with shape of [batch_size, seq_len]
    probs_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    if top_k is None:
        top_k = seq_len
    _, topk_indices = torch.topk(probs, top_k, dim=1)
    probs_mask = probs_mask.scatter_(1, topk_indices, 1)
    probs_mask = torch.where(probs_mask & (probs >= top_p))
    input = input[probs_mask]

    ## third, torch einsum to generate multi-head matmul
    output = torch.einsum("ijh,jhk->ijk", input, weight)

    ## fourth, gradient calculation
    if grad_output == None:
        grad_input, grad_weight = None, None
    else:
        grad_input_ = torch.einsum("ijh,jkh->ijk", grad_output, weight)
        grad_weight = torch.einsum("ijh,ihk->ijk", input.permute(1, 2, 0), grad_output.permute(1, 0, 2))
        grad_input = torch.zeros((batch_size, seq_len, num_heads, head_size))
        grad_input[probs_mask] = grad_input_
        grad_input = grad_input.view(batch_size, seq_len, hidden_size)
        grad_weight = grad_weight.view(hidden_size, embed_size)
    return output, grad_input, grad_weight
    # raise NotImplementedError("TODO: Assignment0 - Task1")