"""
testes_letras - Teste de Compressao de Prompt / Paginacao para o AutoTimes.

Isola o problema de explosao de tokens do pipeline textual (pt-BR por
extenso) comparando 4 templates de formatacao (pt_verboso, pt_curto,
en_curto, json) em 2 modelos do Ablation Study:
    - smetimes: VolLoss + SMETimes + otimizacoes AMD;
    - amd:      somente otimizacoes AMD.

Execucao: python run_testes.py (dentro de testes_letras/).
"""
