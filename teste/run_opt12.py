import sys, os, math, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from benchmark_logger import BenchmarkLogger, Timer
from utils.metrics import metric
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP
from torch.cuda.amp import GradScaler




torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2)
        self.dt_proj = nn.Linear(d_state, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        A = torch.arange(1, d_state + 1).float().repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_half, res = xz.chunk(2, dim=-1)
        x_half = self.conv1d(x_half.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_half = F.silu(x_half)
        delta = F.softplus(self.dt_proj(self.x_proj(x_half).chunk(2, dim=-1)[0]))
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.A_log.shape[0], self.A_log.shape[1], device=x.device)
        y = []
        for t in range(L):
            dA = torch.exp(delta[:, t].unsqueeze(-1) * A.unsqueeze(0))
            h = dA * h + delta[:, t].unsqueeze(-1) * x_half[:, t].unsqueeze(-1)
            y.append((h.sum(-1)) + self.D * x_half[:, t])
        y = torch.stack(y, dim=1)
        y = y * F.silu(res)
        return self.out_proj(y)


class MambaWrapper(nn.Module):
    def __init__(self, d_model): super().__init__(); self.m = MambaBlock(d_model)
    def forward(self, x, *a, **kw): return self.m(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    def forward(self, pos): return (pos[:, None, :].float() * self.inv_freq[None, :, None].float()).cos(), (pos[:, None, :].float() * self.inv_freq[None, :, None].float()).sin()


class MLAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.n_embd; h = config.n_head
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, d // 2)
        self.ke = nn.Linear(d // 2, d); self.ve = nn.Linear(d // 2, d); self.o = nn.Linear(d, d)
        self.rot = RotaryEmbedding(d // h)

    def forward(self, x, *a, **kw):
        B, L, D = x.shape
        head_dim = D // self.q.out_features * D
        q = self.q(x).view(B, L, -1, head_dim)
        kv = self.kv(x)
        k = self.ke(kv).view(B, L, -1, head_dim)
        v = self.ve(kv).view(B, L, -1, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(B, L, D), None


class MoEMLP(nn.Module):
    def __init__(self, d_model, d_ff, n=4, k=2):
        super().__init__(); self.k = k
        self.gate = nn.Linear(d_model, n)
        self.shared = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)) for _ in range(n)])

    def forward(self, x):
        w = F.softmax(self.gate(x), dim=-1)
        tw, ti = torch.topk(w, self.k, dim=-1); tw = tw / (tw.sum(-1, keepdim=True) + 1e-8)
        out = self.shared(x)
        for i in range(self.k):
            for e in range(len(self.experts)):
                m = (ti[:,:,i] == e)
                if m.any(): out[m] += tw[:,:,i][m].unsqueeze(-1) * self.experts[e](x[m])
        return out


class MemoryBank(nn.Module):
    def __init__(self, dim, top_k=3, max_size=1000):
        super().__init__(); self.top_k = top_k; self.max_size = max_size
        self.keys = []; self.vals = []
    def add(self, k, v):
        for a, b in zip(k.detach().cpu(), v.detach().cpu()):
            self.keys.append(a); self.vals.append(b)
        if len(self.keys) > self.max_size: self.keys = self.keys[-self.max_size:]; self.vals = self.vals[-self.max_size:]
    def retrieve(self, q):
        if not self.keys: return None
        q = F.normalize(q, dim=1)
        mk = F.normalize(torch.stack(self.keys).to(q.device), dim=1)
        _, idx = torch.topk(q @ mk.T, min(self.top_k, len(self.keys)), dim=1)
        return torch.stack([torch.stack([self.vals[j].to(q.device) for j in idx[i]]) for i in range(q.shape[0])])


class FinGPTAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self._proj = None
        self.ca = nn.MultiheadAttention(d, 8, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d)
    def forward(self, x, m):
        mark_dim = m.shape[-1]
        if self._proj is None or self._proj.in_features != mark_dim:
            self._proj = nn.Linear(mark_dim, self.d).to(m.device)
        se = self._proj(m.mean(1, keepdim=True)).expand(-1, x.shape[1], -1)
        o, _ = self.ca(x, se, se); return self.norm(o + x)


class MTPDecoder(nn.Module):
    def __init__(self, d, token_len, horizons=[1,2,4]):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d, h * token_len) for h in horizons])
        self.horizons = horizons
    def forward(self, x):
        return {("main" if i==0 else f"aux_{i}"): h(x) for i, h in enumerate(self.heads)}


class GRPOLoss(nn.Module):
    def __init__(self, group=4, alpha=0.1): super().__init__(); self.g = group; self.a = alpha
    def forward(self, pred, target):
        pred = pred.view(self.g, -1, *pred.shape[1:])
        losses = (pred - target.unsqueeze(0)).pow(2).mean(dim=list(range(2, pred.ndim)))
        mean_l = losses.mean(0, keepdim=True); std_l = losses.std(0, keepdim=True) + 1e-8
        adv = (mean_l - losses) / std_l
        return (losses * (1 - self.a * adv.detach())).mean()


class UltimateModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.token_len = configs.token_len
        self.device = f"cuda:{configs.gpu}"
        model_dir = configs.llm_ckp_dir
        if not os.path.isdir(os.path.abspath(model_dir)):
            model_dir = "gpt2"
        self.gpt2 = GPT2Model.from_pretrained(model_dir, device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32, low_cpu_mem_usage=True, attn_implementation="eager")
        self.hidden_dim = self.gpt2.config.n_embd
        for p in self.gpt2.parameters(): p.requires_grad = False

        self.encoder = MoEMLP(configs.token_len, self.hidden_dim*2)
        self.emb_proj = nn.Linear(configs.token_len, self.hidden_dim)
        self._final_proj = None
        self.stat_fusion = nn.Sequential(nn.Linear(4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.stat_gate = nn.Sequential(nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim), nn.Sigmoid())
        self.fingpt = FinGPTAttention(self.hidden_dim)
        self.decoders = MTPDecoder(self.hidden_dim, configs.token_len)
        self.decoder_hidden = MLP(self.hidden_dim, configs.token_len,
            configs.mlp_hidden_dim, configs.mlp_hidden_layers, configs.dropout, configs.mlp_activation)

        h = self.gpt2.config.n_embd; layers = self.gpt2.h
        new_layers = nn.ModuleList()
        for i, layer in enumerate(layers):
            if i == 1 or i == len(layers)-2:
                new_layers.append(MambaWrapper(h))
            elif 0 < i < len(layers)-1 and i % 2 == 1:
                mla = MLAAttention(self.gpt2.config)
                with torch.no_grad():
                    mla.q.weight.copy_(layer.attn.c_attn.weight[:, :h].T)
                    mla.o.weight.copy_(layer.attn.c_proj.weight)
                layer.attn = mla; layer.attn.c_attn = None
                new_layers.append(layer)
            else:
                new_layers.append(layer)
        self.gpt2.h = new_layers

    def forward(self, x_enc, x_mark, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach(); x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5); x_enc = x_enc / stdev
        bs, _, nv = x_enc.shape
        xf = x_enc.permute(0,2,1).reshape(bs*nv, -1)
        fold = xf.unfold(-1, self.token_len, self.token_len)
        tn = fold.shape[1]
        emb = self.encoder(fold)
        emb = self.emb_proj(emb)
        stats = torch.stack([xf.mean(-1), xf.var(-1), xf[:, -1]-xf[:, 0], xf.max(-1).values-xf.min(-1).values], -1)
        sf = self.stat_fusion(stats.unsqueeze(1).expand(-1, tn, -1))
        gate = self.stat_gate(torch.cat([emb, sf], -1))
        emb = gate * emb + (1 - gate) * sf
        if self._final_proj is None or self._final_proj.in_features != emb.shape[-1]:
            self._final_proj = nn.Linear(emb.shape[-1], self.hidden_dim).to(emb.device)
        emb = self._final_proj(emb)
        emb = torch.clamp(emb, -10.0, 10.0)
        out = self.gpt2(inputs_embeds=emb.to(dtype=self.gpt2.dtype).contiguous()).last_hidden_state
        out = out.to(dtype=torch.float32)
        out = self.fingpt(out, x_mark)
        heads = self.decoders(out)
        main = heads["main"].reshape(bs, nv, -1).permute(0,2,1)
        main = main * stdev[:,0,:].unsqueeze(1).expand(-1, tn*self.token_len, -1) + means[:,0,:].unsqueeze(1).expand(-1, tn*self.token_len, -1)
        main = torch.nan_to_num(main, nan=0.0, posinf=1e4, neginf=-1e4)
        return main


class Exp_Ultimate(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self._build_model()
    def _build_model(self):
        m = UltimateModel(self.args).float().to(self.device)
        return m
    def _get_data(self, f): return data_provider(self.args, f)
    def _select_optimizer(self):
        pl = [p for n,p in self.model.named_parameters() if p.requires_grad]
        return torch.optim.Adam([{'params': pl}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
    def _select_criterion(self): return torch.nn.MSELoss()
    def vali(self, dl, crit):
        self.model.eval(); losses = []
        with torch.no_grad():
            for bx, by, bxm, bym in dl:
                bx = bx.float().to(self.device, non_blocking=True)
                bxm = bxm.float().to(self.device, non_blocking=True)
                bym = bym.float().to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    o = self.model(bx, bxm, None, bym)
                    l = crit(o[:,:,:], by[:,:,:].float().to(self.device, non_blocking=True))
                losses.append(l.item())
                if self.args.smoke_test: break
        self.model.train(); return np.mean(losses) if len(losses) > 0 else 0.0
    def train(self, setting):
        td, tl = self._get_data('train'); vd, vl = self._get_data('val'); tsd, tsl = self._get_data('test')
        p = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(p): os.makedirs(p)
        es = EarlyStopping(self.args, verbose=True)
        opt = self._select_optimizer(); crit = self._select_criterion()
        scaler = GradScaler()
        tn = len(tl); t0 = time.time()
        for ep in range(self.args.train_epochs):
            self.model.train(); lv = 0.0
            for i, (bx, by, bxm, bym) in enumerate(tl):
                opt.zero_grad()
                bx = bx.float().to(self.device, non_blocking=True)
                by = by.float().to(self.device, non_blocking=True)
                bxm = bxm.float().to(self.device, non_blocking=True)
                bym = bym.float().to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    o = self.model(bx, bxm, None, bym)
                    l = crit(o, by)
                scaler.scale(l).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(opt); scaler.update()
                if torch.isnan(l) or torch.isinf(l):
                    print(f'  epoch {ep+1} iter {i+1} loss NaN/Inf — skipping')
                else:
                    lv += l.item()
                if self.args.smoke_test: break
                if (i+1)%100==0: print(f'  epoch {ep+1} iter {i+1} loss {l.item():.7f}')
            vl = self.vali(vl, crit)
            print(f'Epoch {ep+1} train {lv/tn:.7f} vali {vl:.7f}')
            es(vl, self.model, p)
            if es.early_stop: break
        self.model.load_state_dict(torch.load(os.path.join(p, 'checkpoint.pth')), strict=False)
    def test(self, setting):
        td, tl = self._get_data('test')
        preds = []; trues = []
        self.model.eval()
        with torch.no_grad():
            for bx, by, bxm, bym in tl:
                bx = bx.float().to(self.device, non_blocking=True)
                bxm = bxm.float().to(self.device, non_blocking=True)
                bym = bym.float().to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    o = self.model(bx, bxm, None, bym)
                preds.append(o[:, -self.args.test_pred_len:, :].cpu())
                trues.append(by[:, -self.args.test_pred_len:, :].cpu())
        p = torch.cat(preds).numpy(); t = torch.cat(trues).numpy()
        mae, mse, rmse, _, _ = metric(p, t)
        print(f'mse:{mse} mae:{mae} rmse:{rmse}')
        return {'mse': mse, 'mae': mae, 'rmse': rmse}


if __name__ == '__main__':
    random.seed(2021); torch.manual_seed(2021); np.random.seed(2021)
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_id', default='opt12'); parser.add_argument('--opt_name', default='Opção Ultimate (Todas as 9)')
    parser.add_argument('--task_name', default='long_term_forecast'); parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', default='test'); parser.add_argument('--model', default='AutoTimes_Gpt2')
    parser.add_argument('--data', default='ETTm1'); parser.add_argument('--root_path', default='./data/ETT/')
    parser.add_argument('--data_path', default='ETTh1.csv'); parser.add_argument('--checkpoints', default='./checkpoints/')
    parser.add_argument('--seq_len', type=int, default=672); parser.add_argument('--label_len', type=int, default=576)
    parser.add_argument('--token_len', type=int, default=96); parser.add_argument('--test_pred_len', type=int, default=96)
    parser.add_argument('--dropout', type=float, default=0.1); parser.add_argument('--llm_ckp_dir', default='./gpt2')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256); parser.add_argument('--mlp_hidden_layers', type=int, default=2)
    parser.add_argument('--mlp_activation', default='tanh'); parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1); parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32); parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001); parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--weight_decay', type=float, default=0); parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--drop_last', action='store_true', default=False)
    parser.add_argument('--val_set_shuffle', action='store_false', default=True)
    parser.add_argument('--drop_short', action='store_true', default=False)
    parser.add_argument('--test_data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--test_seq_len', type=int, default=672)
    parser.add_argument('--test_label_len', type=int, default=576)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--cosine', action='store_true', default=True)
    parser.add_argument('--tmax', type=int, default=10)
    parser.add_argument('--mix_embeds', action='store_true', default=False)
    parser.add_argument('--test_dir', type=str, default='./test')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--smoke_test', type=int, default=0, help='stop after 1 batch per epoch')
    parser.add_argument('--reset_csv', action='store_true', default=False, help='reset benchmark_results.csv before running')
    args = parser.parse_args()

    logger = BenchmarkLogger(reset=args.reset_csv); timer = Timer()
    for ii in range(args.itr):
        setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_ll{args.label_len}_tl{args.token_len}'
        exp = Exp_Ultimate(args)
        timer.start()
        exp.train(setting)
        tt = timer.stop()
        timer.start()
        m = exp.test(setting)
        te = timer.stop()
        total_time = tt + te
        test_loader_len = len(exp._get_data('test')[1])
        avg_latency = te / test_loader_len if test_loader_len > 0 else 0.0
        logger.log(opt_id=args.opt_id, opt_name=args.opt_name, args=args, metrics=m, total_time_s=total_time, avg_latency_s=avg_latency,
                    gpu_time_s=timer.gpu_elapsed, wall_time_s=timer.wall_elapsed)

        print("\n=== Benchmark Results ===")
        print("opt_id:        {}".format(args.opt_id))
        print("opt_name:      {}".format(args.opt_name))
        print("total_time:    {:.4f}s".format(total_time))
        print("test_mse:      {:.8f}".format(m.get('mse', 0)))
        print("test_mae:      {:.8f}".format(m.get('mae', 0)))
        print("test_rmse:     {:.8f}".format(m.get('rmse', 0)))
        print("=========================\n")
        torch.cuda.empty_cache()
