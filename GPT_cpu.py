import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


iteration_count = 0
class EmbeddingLayer(nn.Embedding):
    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp = ninp
        nn.init.uniform_(self.weight, -initrange, initrange)
        self.device = 'cuda:0'
        self.to(self.device)

    def forward(self, src):
        src = src.to(self.device)
        return super().forward(src) * math.sqrt(self.ninp)


class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
        self.device = 'cuda:0'
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
class CacheAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device, dropout=0.1, bias=True, cache_size_limit=0, max_sentence_length=200, max_batch_size=16, use_cache=False):
        super(CacheAttention, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        self.cache_size_limit = cache_size_limit  # 添加缓存大小限制
        self.max_sentence_length = max_sentence_length
        self.max_batch_size = max_batch_size
        self.use_cache = use_cache

        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(device)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(device)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(device)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        original_device = query.device
        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        start_time = time.time()
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, k.size(1)]

        if attn_mask is not None:
            attn_weights += attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        end_time = time.time()

        # 计算并打印消耗的时间
        elapsed_time = end_time - start_time
        # print(f"Attention calculation on CPU took {elapsed_time} seconds")

        return attn





class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, device, dropout=0.1, activation="relu"):
        super(FFN, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(d_model, dim_feedforward).to(device)
        self.linear2 = nn.Linear(dim_feedforward, d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.activation = nn.ReLU().to(device)
        self.norm = nn.LayerNorm(d_model).to(device)
        self.dropout2 = nn.Dropout(dropout).to(device)

    def forward(self, x):
        x = x.to(self.device)
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        out = x + self.dropout2(out)
        out = self.norm(out)
        # out = out.to('cpu')
        return out
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, device, dropout=0.1, activation="relu", use_cache=False):
        super(TransformerDecoderLayer, self).__init__()
        self.device = device
        self.use_cache = use_cache
        self.self_attn = CacheAttention(d_model, nhead, device,dropout=dropout, use_cache=use_cache)
        self.ffn = FFN(d_model, dim_feedforward, device,dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model).to(device)  # Ensure LayerNorm is initialized on the right device
        self.dropout1 = nn.Dropout(dropout).to(device)

    def forward(self, src):
        src = src.to(self.device)
        # src = src.to('cpu')
        src2 = self.self_attn(src, src, src)  # pass src as query, key and value
        src = src + self.dropout1(src2)
        # src = src.to(self.device)
        src = self.norm1(src)  # Now, src tensor is on the same device as LayerNorm weights
        src = self.ffn(src)
        return src



# CacheAttention, FFN, and TransformerDecoderLayer classes would be the same...

class LinearLayer(nn.Linear):
    def __init__(self, ninp, ntoken, initrange):
        super().__init__(ninp, ntoken)
        nn.init.zeros_(self.bias)
        nn.init.uniform_(self.weight, -initrange, initrange)
        self.device = 'cuda:0'
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return super().forward(x)

class TransformerLMSequential(nn.Sequential):
    """A small language model based on the design of GPT-2 using nn.Sequential
    for compatibility with Pipe"""
    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder, use_cache):
        layers = [
            EmbeddingLayer(ntokens, ninp, initrange).to('cuda:0'),  # 假设EmbeddingLayer和PositionalEncodingLayer在同一设备
            PositionalEncodingLayer(ninp, dropout).to('cuda:0'),
        ]
        layers_per_gpu = 8  # 每个GPU有四层
        n_gpus = 1  # 确定GPU数量
        for i in range(ndecoder):
            device = f'cuda:{i // layers_per_gpu % n_gpus}'  # 使用整数商运算符将层分组，并使用模运算符将组分配给GPU
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, device, dropout, use_cache))

        layers.append(LinearLayer(ninp, ntokens, initrange))

        super().__init__(*layers)


class CacheManager:
    def __init__(self, transformer_model):
        self.transformer_model = transformer_model
        self.hooks = []
        self.add_hooks()

    def add_hooks(self):
        def hook(module, input, output):
            # Hook function to be executed after each forward pass of CacheAttention
            cache = module.cache
            # Handle cache updating
            if cache is not None:
                if len(cache[0]) > module.cache_size:
                    # Remove the oldest cache entry
                    cache = (cache[0][1:], cache[1][1:])
                module.cache = cache
            self.hooks.append(module.register_forward_hook(hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_cache(self):
        for module in self.transformer_model.modules():
            if isinstance(module, CacheAttention):
                module.cache = None

    def pad_cache(self, max_length):
        for module in self.transformer_model.modules():
            if isinstance(module, CacheAttention):
                cache = module.cache
                if cache is not None:
                    key, value = cache
                    pad_length = max_length - key.size(0)
                    padded_key = torch.nn.functional.pad(key, (0, 0, 0, pad_length))
                    padded_value = torch.nn.functional.pad(value, (0, 0, 0, pad_length))
                    module.cache = (padded_key, padded_value)

    def crop_cache(self, max_length):
        for module in self.transformer_model.modules():
            if isinstance(module, CacheAttention):
                cache = module.cache
                if cache is not None:
                    key, value = cache
                    cropped_key = key[:max_length]
                    cropped_value = value[:max_length]
                    module.cache = (cropped_key, cropped_value)

    def update_cache(self, input_batch, new_input, new_cache):
        # The new input is placed at the first position in the batched input
        input_batch[0] = new_input

        # We need to align the caches according to the new input
        for module in self.transformer_model.modules():
            if isinstance(module, CacheAttention):
                cache = module.cache
                if cache is not None:
                    key, value = cache
                    new_key, new_value = new_cache
                    # If the new cache is shorter, pad it on the left
                    if new_key.size(1) < key.size(1):
                        print('shorter')
                        pad_len = key.size(1) - new_key.size(1)
                        new_key = torch.nn.functional.pad(new_key, (0, 0, pad_len, 0))
                        new_value = torch.nn.functional.pad(new_value, (0, 0, pad_len, 0))
                    # If the new cache is longer, pad all the other caches on the right
                    elif new_key.size(1) > key.size(1):
                        print('longer')
                        pad_len = new_key.size(1) - key.size(1)
                        key = torch.nn.functional.pad(key, (0, 0, 0, pad_len))
                        value = torch.nn.functional.pad(value, (0, 0, 0, pad_len))
                    else: 
                        print('equal')
                    # Replace the old cache of the first input with the new cache
                    key[0] = new_key[0]
                    value[0] = new_value[0]
                    # Update the cache
                    module.cache = (key, value)