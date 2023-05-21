from typing import Optional, List, Dict, Tuple, Callable, Any
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.autograd import Function

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.0,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        # coeff = float(
        #     2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
        #     - (self.hi - self.lo) + self.lo
        # )
        # coeff = self.iter_num / self.max_iters

        p = float(self.iter_num) / self.max_iters
        coeff = 2. / (1. + np.exp(-10 * p)) - 1

        # coeff = 1

        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

class BERTEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(BERTEncoder, self).__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(config.lm_path)
        for param in self.model.parameters():
            param.requires_grad = True     
        # self.linear = nn.Linear(768, config.encoder_dim)
        
    def forward(self, batch):
        input_ids, token_type_ids, attention_mask = batch['input_ids'], batch['token_type_ids'], batch['attention_mask']
        # last hidden_state? -> size (batch_size, seq_len, hidden_size)
        hidden_states = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        # pooled_output -> size (batch_size, 1, hidden_size) -> (batch_size, hidden_size)
        pooled_output = nn.functional.avg_pool2d(hidden_states, kernel_size=(self.config.seq_len, 1)).squeeze(1)
        return pooled_output # (batch_size, config.encoder_dim)
    

class RegressorWithDANN(nn.Module):
    def __init__(self, encoder, config) -> None:
        super(RegressorWithDANN, self).__init__()
        self.config = config
        self.encoder = encoder
        self.encoder_dim = config.encoder_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder_dim, 1),
            nn.Sigmoid(),
        )
        # self.discriminator = nn.Sequential(
        #     nn.Linear(self.encoder_dim, 200),
        #     nn.BatchNorm1d(200),
        #     nn.ReLU(),
        #     nn.Linear(200, 2),
        # )
        self.discriminator = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_dim, 2),
        )
        self.grl = WarmStartGradientReverseLayer(max_iters=config.max_iters)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, batch_s, batch_t):
        embedding_s = self.dropout(self.encoder(batch_s))  # (batch_size, self.encoder_dim)
        embedding_t = self.dropout(self.encoder(batch_t))
        score_s = self.regressor(embedding_s)

        domain_s = self.discriminator(self.grl(embedding_s))
        domain_t = self.discriminator(self.grl(embedding_t))

        return score_s, domain_s, domain_t
    
    def inference(self, batch):
        return self.regressor(self.encoder(batch))
    
    def step(self):
        self.grl.iter_num += 1

    def get_parameters(self, base_lr=1e-5) -> List[Dict]:
        params = [
            {"params": self.encoder.parameters(), "lr": base_lr},
            {"params": self.regressor.parameters(), "lr": base_lr * 100},
            {"params": self.discriminator.parameters(), "lr": base_lr * 100},
        ]
        return params
    

    


        
