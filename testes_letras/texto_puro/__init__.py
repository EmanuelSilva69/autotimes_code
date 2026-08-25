"""
texto_puro - Avaliacao de tokenizacao e paginacao (chunking) de textos puros.

Problema: o GPT-2 tem limite rigido de contexto e vocabulario nativo em
ingles; textos longos (logs, noticias, descricoes) em portugues agravam
lentidao, estouro de memoria e underfitting.

Este modulo mede, para os MESMOS documentos em PT-BR e EN:
    1) INFLACAO DE TOKENS: quantos tokens o GPT-2 gera em PT vs EN;
    2) PAGINACAO (chunking): 3 estrategias para caber no contexto -
       truncamento direto, janela deslizante com sobreposicao e
       compressao por palavras-chave (remoção de stopwords);
    3) ESTABILIDADE DA LOSS: o texto pago alimenta o ramo textual da
       mesma arquitetura aprimorada (VolLoss + SMETimes) e medimos
       convergencia, tempo por batch e batches estourados (NaN/Inf).

Execucao: python -m testes_letras.run_texto (na raiz do projeto).
"""
