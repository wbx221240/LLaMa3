import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import RMSNorm


## Group RMSNorm block
"""
This RMSNorm version resembles the multi-head adaptation of attention mechanism.
- first, what we need for a common class?
    - a learnable re-scaling matrix gamma with shape of [hidden_size, ]
    - parameter resetting method
- second, for a grouped version of RMSNorm:
    - the learnable parameter should be shape of [group_num, group_size]
    - parameter resetting method

"""


class GroupRMSNorm(nn.Module):
    """Group RMS Norm module
    This is a variant of RMS Norm that \
        evenly splits the hidden dimension into groups, and \
        applies root-mean-square normalization with \
            learnable scaling transformation on each i-th group individually.
    """
    
    def __init__(self, 
        hidden_size: int, 
        group_size: int,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize Group RMS Norm module
        
        Args:
            hidden_size(int): hidden dimension size
            group_size(int): group size
            eps(float, default = 1e-5): epsilon
            init_range(tuple, default = (-1.0, 1.0)): the range of the uniform distribution to initialize learnable scaling parameters
            init_seed(int, default = 42): seed for the initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.group_size = group_size
        # self.group_nums = hidden_size // group_size
        factory_kwargs = {"device": device, "dtype": dtype}
        # print(factory_kwargs)
        self.init_range = init_range
        self.init_seed = init_seed
        self.eps = eps
        self.weight = Parameter(torch.empty((self.hidden_size // self.group_size, self.group_size), **factory_kwargs))
        self.reset_parameters()
        # raise NotImplementedError("TODO: Assignment1 - Task1")
        
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        """The forward pass for Group RMS Norm module

        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): normalized output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        # raise NotImplementedError("TODO: Assignment1 - Task1")
        ## first, split the input into groups along the hidden dimension
        batch_size, seq_len, _ = input.shape
        input_type = input.dtype
        input = input.to(torch.float32)
        input = input.view(batch_size, seq_len, -1, self.group_size)
        rms_norm = torch.mean(input.pow(2), dim=-1, keepdim=True)# .repeat_interleave(self.group_size, dim=-1).reshape(batch_size, seq_len, -1, self.group_size)
        # print(rms_norm.shape)
        # assert 0
        normed_input = input * torch.rsqrt(rms_norm + self.eps) 
        # normed_input = normed_input.reshape(-1, self.group_size, batch_size, seq_len).permute(2, 3, 0, 1)
        return (normed_input * self.weight).reshape(batch_size, seq_len, -1).to(input_type)
        ## apply RMSNorm in each groups
    
    def reset_parameters(self) -> None:
        """Initialize learnable scaling parameters for Group RMS Norm from a uniform distribution"""
        torch.manual_seed(self.init_seed)
        nn.init.uniform_(self.weight, a=self.init_range[0], b=self.init_range[1])
        # raise NotImplementedError("TODO: Assignment1 - Task1")

