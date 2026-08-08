import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


class RAGRetriever(nn.Module):
    def __init__(self, key_dim, top_k=3, max_memory_size=1000):
        super().__init__()
        self.key_dim = key_dim
        self.top_k = top_k
        self.max_memory_size = max_memory_size
        self._keys = []
        self._vals = []

    def add(self, keys, values):
        for k, v in zip(keys.detach().cpu(), values.detach().cpu()):
            self._keys.append(k)
            self._vals.append(v)
        if len(self._keys) > self.max_memory_size:
            self._keys = self._keys[-self.max_memory_size:]
            self._vals = self._vals[-self.max_memory_size:]

    def retrieve(self, query):
        if not self._keys:
            return None
        query = F.normalize(query, dim=1)
        mem_keys = F.normalize(torch.stack(self._keys).to(query.device), dim=1)
        sim = query @ mem_keys.T
        topk = min(self.top_k, len(self._keys))
        _, topk_idx = torch.topk(sim, topk, dim=1)
        device = query.device
        batch_out = []
        for b in range(query.shape[0]):
            retrieved = torch.stack([self._vals[idx].to(device) for idx in topk_idx[b]])
            batch_out.append(retrieved)
        return torch.stack(batch_out)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        model_dir = configs.llm_ckp_dir
        if not os.path.isdir(os.path.abspath(model_dir)):
            model_dir = "gpt2"
        self.gpt2 = GPT2Model.from_pretrained(
            model_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.hidden_dim_of_gpt2 = self.gpt2.config.n_embd
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation)

        self.encoder.to(dtype=self.gpt2.dtype)
        self.decoder.to(dtype=self.gpt2.dtype)

        self.retriever = RAGRetriever(
            key_dim=self.hidden_dim_of_gpt2,
            top_k=getattr(configs, 'rag_top_k', 3),
            max_memory_size=getattr(configs, 'rag_memory_size', 1000),
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc = x_enc.reshape(bs * n_vars, -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        fold_out = fold_out.to(dtype=self.gpt2.dtype)
        times_embeds = self.encoder(fold_out).to(dtype=self.gpt2.dtype)
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        rag_retrieved = self.retriever.retrieve(times_embeds.mean(dim=1))
        if rag_retrieved is not None:
            B, TK, TN, HD = rag_retrieved.shape
            rag_retrieved = rag_retrieved.reshape(B, TK * TN, HD)
            rag_retrieved = rag_retrieved.to(dtype=self.gpt2.dtype)
            times_embeds = torch.cat([rag_retrieved, times_embeds], dim=1)
            times_embeds = times_embeds.to(dtype=self.gpt2.dtype)

        outputs = self.gpt2(inputs_embeds=times_embeds.to(dtype=self.gpt2.dtype).contiguous()).last_hidden_state

        if rag_retrieved is not None:
            outputs = outputs[:, -token_num:, :]

        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))

        return dec_out

    def add_to_memory(self, x_enc, x_mark_enc):
        with torch.no_grad():
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            bs, _, n_vars = x_enc.shape
            x_enc = x_enc.permute(0, 2, 1).contiguous()
            x_enc = x_enc.reshape(bs * n_vars, -1)
            fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
            fold_out = fold_out.to(dtype=self.gpt2.dtype)
            times_embeds = self.encoder(fold_out).to(dtype=self.gpt2.dtype)
            if self.mix:
                times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
                x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
                times_embeds = times_embeds + self.add_scale * x_mark_enc
            keys = times_embeds.mean(dim=1)
            self.retriever.add(keys, times_embeds)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
