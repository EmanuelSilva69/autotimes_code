"""
idioma - Avaliacao de Idioma e Tokenizacao (PT vs EN).

Como o GPT-2 foi treinado nativamente em ingles, o mesmo conteudo em
pt-BR fragmenta palavras em mais sub-tokens. Este modulo calcula a
"Taxa de Inflacao de Tokens":

    taxa_inflacao = tokens_gpt2(texto_pt) / tokens_gpt2(texto_en)

para pares de documentos equivalentes (mesmo conteudo, dois idiomas),
com agregados por documento e taxa media do corpus.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def medir_inflacao_tokenizacao(
    pares: List[Tuple[str, str, str]],
    tokenizer,
) -> Dict:
    """Mede a inflacao de tokens PT vs EN de um corpus pareado.

    Args:
        pares: lista (nome, texto_pt, texto_en) - ver amostras.carregar_pares.
        tokenizer: tokenizer GPT-2 local.

    Returns:
        Dict com taxa_media_inflacao, tokens totais por idioma e a
        lista por_documento (nome, tokens_pt, tokens_en, taxa).
    """
    linhas: List[Dict] = []
    for nome, pt, en in pares:
        ids_pt = tokenizer(pt, add_special_tokens=False)["input_ids"]
        ids_en = tokenizer(en, add_special_tokens=False)["input_ids"]
        n_pt, n_en = len(ids_pt), len(ids_en)
        linhas.append({
            "documento": nome,
            "tokens_pt": n_pt,
            "tokens_en": n_en,
            "taxa_inflacao": round(n_pt / max(n_en, 1), 4),
        })

    total_pt = sum(l["tokens_pt"] for l in linhas)
    total_en = sum(l["tokens_en"] for l in linhas)
    return {
        "taxa_media_inflacao": round(total_pt / max(total_en, 1), 4),
        "tokens_totais_pt": total_pt,
        "tokens_totais_en": total_en,
        "por_documento": linhas,
    }
