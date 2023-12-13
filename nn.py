import torch
import torch.nn as nn
from typing import List, Callable, Optional
from transformers import AutoTokenizer, T5EncoderModel

class WLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias_size: Optional[int] = None
    ):
        super().__init__()

        if bias_size is None:
            bias_size = out_features

        dim = 1000
        self.z = nn.Parameter(torch.empty(dim).normal_(0, 1.0 / out_features))
        self.fc = nn.Linear(dim, in_features * out_features + out_features)
        self.seq = self.fc
        self.w_idx = in_features * out_features
        self.weight = self.fc.weight
        self._linear = self.fc
        self.out_f = out_features

    def adaptation_parameters(self):
        return [self.z]

    def forward(self, x: torch.tensor):
        theta = self.fc(self.z)
        w = theta[: self.w_idx].view(x.shape[-1], -1)
        b = theta[self.w_idx :]

        ret = x @ w + b
        # import pdb; pdb.set_trace()
        return x @ w + b


class MLP(nn.Module):
    def __init__(
        self,
        layer_widths: List[int],
        final_activation: Callable = lambda x: x,
        bias_linear: bool = False,
        extra_head_layers: List[int] = None,
        w_linear: bool = False,
        llm_flag = False,
    ):
        super().__init__()

        if len(layer_widths) < 2:
            raise ValueError(
                "Layer widths needs at least an in-dimension and out-dimension"
            )

        self._final_activation = final_activation
        self.seq = nn.Sequential()
        self._head = extra_head_layers is not None
        self.llm_flag = llm_flag

        # if not w_linear:
        #     linear = BiasLinear if bias_linear else nn.Linear
        # else:
        #     linear = WLinear
        # self.bias_linear = bias_linear
        linear = WLinear
        self.aparams = []
        self.descript_transform = None
        self.fusion_transform = None

        if self.llm_flag:
            self.descript_transform = nn.Linear(100*512, 128)
            self.fusion_transform = nn.Linear(256, 128)

        for idx in range(len(layer_widths) - 1):
            w = linear(layer_widths[idx], layer_widths[idx + 1])
            self.seq.add_module(f"fc_{idx}", w)
            if idx < len(layer_widths) - 2:
                self.seq.add_module(f"relu_{idx}", nn.ReLU())

        if extra_head_layers is not None:
            self.pre_seq = self.seq[:-2]
            self.post_seq = self.seq[-2:]

            self.head_seq = nn.Sequential()
            extra_head_layers = [
                layer_widths[-2] + layer_widths[-1]
            ] + extra_head_layers

            for idx, (infc, outfc) in enumerate(
                zip(extra_head_layers[:-1], extra_head_layers[1:])
            ):
                self.head_seq.add_module(f"relu_{idx}", nn.ReLU())
                w = linear(extra_head_layers[idx], extra_head_layers[idx + 1])
                self.head_seq.add_module(f"fc_{idx}", w)

        # import pdb; pdb.set_trace()


    def forward(self, x: torch.tensor, acts: Optional[torch.tensor] = None, descript_vector=None):
        if self._head and acts is not None:
            h = self.pre_seq(x)
            # print("!!!FORWARD!!!")
            # print("H:", h.shape, h)
            # print("ACTS:", acts.shape, acts)
            if descript_vector != None:
                trans_descript_vector = self.descript_transform(descript_vector.view(1, -1))
                post_h = torch.cat((h, trans_descript_vector.repeat(h.shape[0],1)), dim=1)
                post_h = self.fusion_transform(post_h)
                
            head_input = torch.cat((h, acts), -1)
            # print("Head Input:", head_input.shape)
            # print("Head Seq:", self.head_seq(head_input))
            
            if torch.isnan(self._final_activation(self.post_seq(h))[0][0]): 
                import pdb; pdb.set_trace()
            if descript_vector != None:
                return self._final_activation(self.post_seq(post_h)), self.head_seq(head_input)  
            else:
                return self._final_activation(self.post_seq(h)), self.head_seq(head_input)
        else:
            if descript_vector != None:
                h = self.pre_seq(x)
                trans_descript_vector = self.descript_transform(descript_vector.view(1, -1))
                post_h = torch.cat((h, trans_descript_vector.repeat(h.shape[0],1)), dim=1)
                post_h = self.fusion_transform(post_h)

                return self._final_activation(self.post_seq(post_h))
            else:
                return self._final_activation(self.seq(x))

class LLM_MLP(nn.Module):
    def __init__(self, obs_dim, args, action_dim, policy_head, flag):
        super().__init__()
        self.mlp = MLP(
            [obs_dim] + [args.net_width] * args.net_depth + [action_dim],
            final_activation=torch.tanh,
            extra_head_layers=policy_head,
            w_linear=args.weight_transform,
            llm_flag=flag
        ) if flag == 'policy' else MLP(
            [obs_dim] + [args.net_width] * args.net_depth + [1],
            w_linear=args.weight_transform,
        )
        self.llm_encoder = T5EncoderModel.from_pretrained("t5-small")
        self.llm_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model_freeze(self.llm_encoder)

    def forward(self, x, acts=None, description=None):
        input_ids = self.llm_tokenizer(description[0], return_tensors="pt", padding='max_length', max_length=100, truncation=True,).input_ids.to('cuda') # Batch size 1
        outputs = self.llm_encoder(input_ids=input_ids)
        descript_vector = outputs.last_hidden_state.squeeze()
        
        return self.mlp(x, acts, descript_vector)
    
    def model_freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

if __name__ == "__main__":
    mlp = MLP([1, 5, 8, 2])
    x = torch.empty(10, 1).normal_()
    print(mlp(x).shape)