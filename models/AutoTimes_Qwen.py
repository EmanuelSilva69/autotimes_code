import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from layers.mlp import MLP

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        torch.set_float32_matmul_precision('high' if configs.use_amp else 'highest')
        torch.backends.cudnn.benchmark = configs.use_amp
        self.use_amp = configs.use_amp

        self.token_len = configs.token_len
        
        # Gerenciamento de GPU
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        
        self.qwen = AutoModelForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        
        # [MELHORIA] Lê a dimensão dinamicamente em vez de fixar (suporta 0.5B, 1.5B, etc.)
        self.hidden_dim_of_qwen = self.qwen.config.hidden_size
        
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        
        # Congela os pesos do LLM (apenas o encoder/decoder linear será treinado)
        for name, param in self.qwen.named_parameters():
            param.requires_grad = False

        # Constrói os projetores lineares baseados na dimensão extraída
        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_qwen)
            self.decoder = nn.Linear(self.hidden_dim_of_qwen, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_qwen,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_qwen, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation)

    def forward(self, x_enc, x_mark_enc,x_dec=None, x_mark_dec=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            bs, seq_len, n_vars = x_enc.shape
            
            # Alinhamento das dimensões
            x_enc = x_enc.permute(0, 2, 1)
            x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
            
            # Criação dos Tokens (Patching)
            fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
            token_num = fold_out.shape[1]
            
            # Projeção temporal (Embedding)
            times_embeds = self.encoder(fold_out)
            
            if self.mix:
                times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
                x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
                times_embeds = times_embeds + self.add_scale * x_mark_enc
                
            # Passagem pelo "Cérebro" do LLM (Qwen)
            outputs = self.qwen.model(inputs_embeds=times_embeds)[0]
            
            # Decodificação para a previsão
            dec_out = self.decoder(outputs)
            dec_out = dec_out.reshape(bs, n_vars, token_num, self.token_len)
            dec_out = dec_out.reshape(bs, n_vars, -1)
            dec_out = dec_out.permute(0, 2, 1)
            
            return dec_out