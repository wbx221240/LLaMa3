from enum import Enum
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.signal.windows import general_cosine


class BiLinearActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = None

    def forward(self, input):
        self.act = torch.ones_like(input)
        return input

class MLPActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    BILINEAR = "bilinear"

ACT2CLS = {
    MLPActivationType.RELU: F.relu,
    MLPActivationType.GELU: F.gelu,
    MLPActivationType.SILU: F.silu,
    MLPActivationType.SIGMOID: F.sigmoid,
    MLPActivationType.BILINEAR: BiLinearActivation()
}



class DenseMLPWithLoRA(nn.Module):
    """Dense MLP module with LoRA adapters
    This is a GLU-style dense MLP layer with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Dense MLP module with LoRA adapters
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): intermediate dimension size
            activation_type(MLPActivationType, default = "silu"): activation type, it is a instance of MLPActivationType with corresponding acti name.
            init_base_seed(int, default = 42): seed for base weight initialization
            lora_rank(int, default = 0): lora rank, if 0, then no lora to apply
            lora_alpha(Optional[float], default = None): lora alpha, if None, then set to lora_rank
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.ffh_size = ffh_size
        self.activation = ACT2CLS[activation_type]
        self.activation_type = activation_type

        self.init_base_seed = init_base_seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_rank if lora_alpha is None else lora_alpha
        self.lora_dropout_rate = lora_dropout_rate
        self.lora_dropout_seed = lora_dropout_seed
        self.lora_init_base_seed = lora_init_base_seed
        self.factory_kwargs = {"dtype": dtype, "device": device}
        
        ## proj matrix 
        # self.up_proj = nn.Linear(hidden_size, ffh_size, bias=False, dtype=dtype, device=device)
        # self.gate_proj = nn.Linear(hidden_size, ffh_size, bias=False, dtype=dtype, device=device)
        # self.down_proj = nn.Linear(ffh_size, hidden_size, bias=False, dtype=dtype, device=device)

        self.up_proj = nn.Parameter(torch.zeros(hidden_size, ffh_size, **self.factory_kwargs), requires_grad=True)
        self.gate_proj = nn.Parameter(torch.zeros(hidden_size, ffh_size, **self.factory_kwargs), requires_grad=True)
        self.down_proj = nn.Parameter(torch.zeros(ffh_size, hidden_size, **self.factory_kwargs), requires_grad=True)
        self.lora_dropout = nn.Dropout(p=self.lora_dropout_rate)

        ## LoRA parameters
        if self.lora_rank:
            self.lora_A = nn.Parameter(torch.zeros(hidden_size, lora_rank, dtype=dtype, device=device), requires_grad=True)
            self.lora_B = nn.Parameter(torch.zeros(lora_rank, hidden_size, dtype=dtype, device=device), requires_grad=True)

        self.reset_parameters()
        # print("lora A:", self.lora_A)
        # print("lora_B:", self.lora_B)


        
        # raise NotImplementedError("Assignment2 - Task1")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Dense MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        ## MLP(x) = (\phi(X \times gate) \odot X \times up) \times down
        dtype, device = input.dtype, input.device
        input = input.to(self.factory_kwargs["dtype"]).to(self.factory_kwargs["device"])
        # gated = self.gate_proj(input)
        # up = self.up_proj(input)
        # output = self.down_proj(self.activation(gated) * up)
        # output = self.down_proj(self.activation(self.gate_proj(input)) * self.up_proj(input))
        output = (self.activation(input @ self.gate_proj) * (input @ self.up_proj)) @ self.down_proj
        if self.lora_rank:
            # print("Use LoRA")
            torch.manual_seed(self.lora_dropout_seed)
            lora_term = (input @ self.lora_A @ self.lora_B) * (self.lora_alpha / self.lora_rank)
            # dropout = self.lora_dropout(self.lora_alpha / self.lora_rank * (input @ self.lora_A @ self.lora_B))
            output = output + self.lora_dropout(lora_term)
        return output.to(dtype).to(device)
        
        # raise NotImplementedError("Assignment2 - Task1")
    
    def reset_parameters(self):
        """Initialize the weights of the Dense MLP module with LoRA adapters
        from a normal distribution (or a uniform distribution for lora weights)
        """
        ## initialization of GLU-type of MLP
        if self.activation_type in [MLPActivationType.SIGMOID, MLPActivationType.BILINEAR]:
            
            nn.init.xavier_normal_(self.up_proj.T, generator=torch.manual_seed(self.init_base_seed + 1))
            
            nn.init.xavier_normal_(self.gate_proj.T, generator=torch.manual_seed(self.init_base_seed + 2))
            
            nn.init.xavier_normal_(self.down_proj.T, generator=torch.manual_seed(self.init_base_seed + 3))

        else:
            # print(self.activation_type)
            nn.init.kaiming_normal_(self.up_proj.T, mode='fan_in', nonlinearity="relu", generator=torch.manual_seed(self.init_base_seed + 1))
            
            nn.init.kaiming_normal_(self.gate_proj.T, mode="fan_in", nonlinearity="relu", generator=torch.manual_seed(self.init_base_seed + 2))
            
            nn.init.kaiming_normal_(self.down_proj.T, mode="fan_in", nonlinearity="relu", generator=torch.manual_seed(self.init_base_seed + 3))
        
        ## initialization of LoRA
        if self.lora_rank:
            nn.init.kaiming_uniform_(self.lora_A.T, mode="fan_in", nonlinearity="relu", generator=torch.manual_seed(self.lora_init_base_seed + 1))
            nn.init.kaiming_uniform_(self.lora_B.T, mode="fan_in", nonlinearity="relu", generator=torch.manual_seed(self.lora_init_base_seed + 2))
        # raise NotImplementedError("Assignment2 - Task1")

    
class SparseMLPWithLoRA(nn.Module):
    """Sparse MLP module with LoRA adapters
    This is a GLU-style sparse MLP layer with LoRA adapters, \
        where the sparcity is implemented as Mixture of Experts (MoE), \
            and each expert is a dense MLP with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        num_experts: int = 1,
        moe_topk: int = 1,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Sparse MLP module with LoRA adapters
        
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = MLPActivationType.SILU): activation type
            num_experts(int, default = 1): number of (global) experts, which can deduce expert_size = ffh_size // num_experts
            moe_topk(int, default = 1): topk-routing for MoE to control the sparcity
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean for the initialization
            init_std(float, default = 1.0): std for the initialization
            init_base_seed(int, default = 42): seed for the initialization
            lora_rank(int, default = 0): lora rank
            lora_alpha(Optional[float], default = None): lora alpha
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        assert not ffh_size % num_experts, "ffh size not divisible by num experts"
        assert not num_experts % world_size, "num experts not divisible by world_size"

        self.hidden_size = hidden_size
        self.ffh_size = ffh_size
        self.activation = ACT2CLS[activation_type]
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.rank = rank
        self.world_size = world_size
        self.init_mean = init_mean
        self.init_std = init_std
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout_rate = lora_dropout_rate
        
        self.factory_kwargs = {"dtype": dtype, "device":device}
        
        self.nle = self.num_experts // self.world_size # this is the number of experts in each rank.
        self.init_gate_seed = init_base_seed
        self.init_base_seed = init_base_seed + self.rank * self.nle
        self.lora_dropout_seed = lora_dropout_seed + self.rank * self.nle
        self.lora_init_base_seed = lora_init_base_seed + self.rank * self.nle

        self.gate = nn.Parameter(torch.zeros((hidden_size, num_experts), dtype=torch.float32, device=device))

        ## create a module list for experts
        self.experts = nn.ModuleList([DenseMLPWithLoRA(hidden_size, ffh_size // num_experts, \
            activation_type, self.init_base_seed + i, lora_rank, lora_alpha, lora_dropout_rate, self.lora_dropout_seed + i, \
            self.lora_init_base_seed + i, dtype, device) for i in range(0, self.nle)])
        
        self.reset_parameters()
        # raise NotImplementedError("Assignment2 - Task2")
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Sparse MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = input.shape
        output = torch.zeros((batch_size * seq_len, hidden_size), dtype=self.gate.dtype, device=input.device)

        input = input.view(-1, hidden_size)
        input_dtype = input.dtype
        input = input.to(self.gate.dtype)
        router_logits = torch.matmul(input, self.gate)
        print(input.shape)
        print(input)

        router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        router_weights, selected_gates = torch.topk(router_weights, k=self.moe_topk, dim=-1)
        router_weights /= router_weights.sum(dim=-1, keepdim=True)
        print(router_weights.shape)
        print(router_weights)
        print(selected_gates)
        expert_mask = F.one_hot(selected_gates, num_classes=self.num_experts).permute(2, 1, 0)
        print(expert_mask.shape)
        print(expert_mask)
        for expert_id in range(self.nle):
            expert = self.experts[expert_id]
            idx, top_x = torch.where(expert_mask[self.rank * self.nle + expert_id])

            current_state = input[None, top_x].reshape(-1, hidden_size)
            partial_output = expert(current_state) * router_weights[top_x, idx, None]

            output.index_add_(0, top_x, partial_output.to(output.dtype))

        output = output.view(batch_size, seq_len, hidden_size)
        return output.to(input_dtype)
        # raise NotImplementedError("Assignment2 - Task2")
        
    def reset_parameters(self):
        """Initialize the weights of each local expert from its own distribution \
            and the gating layer from a normal distribution
        """
        ## initialize the gating matrix.
        nn.init.normal_(self.gate, self.init_mean, self.init_std, generator=torch.manual_seed(self.init_gate_seed))
        # raise NotImplementedError("Assignment2 - Task2")