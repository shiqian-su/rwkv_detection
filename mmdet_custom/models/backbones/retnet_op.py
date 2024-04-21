import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class RetNetRelPos(nn.Module):
    def __init__(self, embed_dim, args):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // args.vir_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(args.vir_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        # self.recurrent_chunk_size = args.recurrent_chunk_size
        
    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            
            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (scale / mask[:, -1].sum(dim=-1)[:, None, None])
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay, value_inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            # mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            # mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.abs(index[:, None] - index[None, :])
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


class RetNetRelPosDeform(nn.Module):
    def __init__(self, embed_dim, args):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // args.vir_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(args.vir_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        # self.recurrent_chunk_size = args.recurrent_chunk_size
        
    def forward(self, slen, offsets):
        B, H, L  = offsets.shape
        index = torch.arange(slen).to(self.decay)
        sin_k = torch.sin(index[:, None] * self.angle[None, :])
        cos_k = torch.cos(index[:, None] * self.angle[None, :])
        sin_q = torch.sin((index+offsets).view(B, H, L, 1) * self.angle.view(1, 1, 1, -1))
        cos_q = torch.cos((index+offsets).view(B, H, L, 1) * self.angle.view(1, 1, 1, -1))
        # mask = torch.tril(torch.ones(slen, slen).to(self.decay))
        # mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
        mask = torch.abs((index+offsets).view(B, H, L, 1) - index.view(1, 1, 1, -1))
        mask = torch.exp(mask * self.decay[None, :, None, None])
        mask = torch.nan_to_num(mask)
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
        retention_rel_pos = ((sin_q, cos_q, sin_k, cos_k), mask)

        return retention_rel_pos


class RetNetRelPosDeform2D(nn.Module):
    def __init__(self, embed_dim, args):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // args.vir_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-2 - 5*torch.arange(args.vir_heads, dtype=torch.float)/args.vir_heads))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        # self.recurrent_chunk_size = args.recurrent_chunk_size
        
    def forward(self, slen, offsets):
        B, H, N1, N2, _  = offsets.shape
        index1 = torch.arange(N1).to(self.decay)
        index2 = torch.arange(N2).to(self.decay)
        index_k = (index1.view(-1, 1) + index2.view(1, -1)).view(slen)
        sin_k = torch.sin(index_k[:, None] * self.angle[None, :])
        cos_k = torch.cos(index_k[:, None] * self.angle[None, :])

        index1_q = index1.view(1, 1, N1, 1)+offsets[..., 0]
        index2_q = index2.view(1, 1, 1, N2)+offsets[..., 1]
        index_q = index1_q + index2_q
        sin_q = torch.sin((index_q).view(B, H, slen, 1) * self.angle.view(1, 1, 1, -1))
        cos_q = torch.cos((index_q).view(B, H, slen, 1) * self.angle.view(1, 1, 1, -1))
        # mask = torch.tril(torch.ones(slen, slen).to(self.decay))
        # mask = torch.masked_fill(index_k[:, None] - index_k[None, :], ~mask.bool(), float("inf"))
        mask = (torch.abs(index1_q.view(B, H, N1, N2, 1, 1) - index1.view(1, 1, 1, 1, N1, 1)) + \
            torch.abs(index2_q.view(B, H, N1, N2, 1, 1) - index2.view(1, 1, 1, 1, 1, N2))).reshape(B, H, slen, slen)
        mask = torch.exp(mask * self.decay[None, :, None, None])
        mask = torch.nan_to_num(mask)
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
        retention_rel_pos = ((sin_q, cos_q, sin_k, cos_k), mask)

        return retention_rel_pos
    

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


class to_channels_first(nn.Module):

    def __init__(self, dim=3):
        super().__init__()
        self.dim=dim

    def forward(self, x):
        if self.dim == 4:
            return x.permute(0, 3, 1, 2)
        if self.dim == 3:
            return x.permute(0, 2, 1)


class to_channels_last(nn.Module):

    def __init__(self, dim=3):
        super().__init__()
        self.dim=dim

    def forward(self, x):
        if self.dim == 4:
            return x.permute(0, 2, 3, 1)
        if self.dim == 3:
            return x.permute(0, 2, 1)


class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        value_dim,
        num_heads,
        gate_fn="swish",
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        
        # self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        # self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)
        
        self.ln = nn.LayerNorm(self.head_dim)
        self.gelu = nn.GELU()
        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)

        if args.vir_use_pos == '2d':
            self.rel_pos = RetNetRelPosDeform2D(embed_dim, args)
            if args.vir_use_deformable:
                self.offset = nn.Sequential(
                    to_channels_first(dim=4),
                    nn.Conv2d(
                        embed_dim,
                        embed_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=embed_dim),
                    to_channels_last(dim=4),
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    nn.GELU(),
                    nn.Linear(
                        embed_dim,
                        num_heads*2)
                    )
        elif args.vir_use_pos == '1d':
            self.rel_pos = RetNetRelPosDeform(embed_dim, args)
            if args.vir_use_deformable:
                self.offset = nn.Sequential(
                    to_channels_first(dim=3),
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=embed_dim),
                    to_channels_last(dim=3),
                    nn.LayerNorm(embed_dim, eps=1e-6),
                    nn.GELU(),
                    nn.Linear(
                        embed_dim,
                        num_heads)
                    )

        # self.group_norm = RMSNorm(self.head_dim, eps=args.layernorm_eps, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        # nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -1)
        if self.args.vir_use_deformable:
            nn.init.constant_(self.offset[5].weight, 0)
            nn.init.constant_(self.offset[5].bias, 0)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output
    
    def forward(
        self,
        x,
        chunkwise_recurrent=False,
        incremental_state=None
    ):
        bs, N1, N2, Cin = x.shape
        x = x.view(bs, N1*N2, Cin)
        bsz, tgt_len, _ = x.size()

        if self.args.vir_use_pos == '2d':
            N1, N2 = int(math.sqrt(tgt_len)), int(math.sqrt(tgt_len))
            offsets = torch.zeros((bsz, self.num_heads, N1, N2, 2)).to(x)
            if self.args.vir_use_deformable:
                offset_x = x.view(bsz, N1, N2, -1)
                offsets = self.offset(offset_x).view(bsz, N1, N2, self.num_heads, 2)
                offsets = offsets.permute(0, 3, 1, 2, 4)
            (sin_q, cos_q, sin_k, cos_k), inner_mask = self.rel_pos(tgt_len, offsets)
        elif self.args.vir_use_pos == '1d':
            offsets = torch.zeros((bsz, self.num_heads, tgt_len)).to(x)
            if self.args.vir_use_deformable:
                offsets = self.offset(x).view(bsz, tgt_len, self.num_heads).transpose(1, 2)
            (sin_q, cos_q, sin_k, cos_k), inner_mask = self.rel_pos(tgt_len, offsets)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

        qr = theta_shift(q, sin_q, cos_q)
        kr = theta_shift(k, sin_k, cos_k)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)
        
        # output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        # output = self.gate_fn(g) * output
        output = self.ln(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)
        output = self.gelu(output)
        # sampling = output.permute(0,2,1).contiguous().view(bsz, -1, N1, N2)
        output = self.out_proj(output)

        output = output.view(bs, N1, N2, Cin)
        sampling = output.permute(0,3,1,2)
        if self.args.MODEL.draw_grad:
            return output, None, sampling
        return output