#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Download Automatizado do Acervo Científico AutoTimes (Embeddings & Causalidade)
Baixa os 49 artigos catalogados diretamente do arXiv em subpastas temáticas,
valida a integridade dos PDFs e gera um manifesto CSV completo.
"""

import csv
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

# Diretório base onde os artigos serão salvos
BASE_DIR = Path("pesquisas/acervo_arxiv_autotimes_2026")

# Catálogo completo de 49 artigos: (arxiv_id, pasta_categoria, nome_curto, titulo_descritivo)
CATALOGO_ARTIGOS: List[Tuple[str, str, str, str]] = [
    # -------------------------------------------------------------------------
    # 01. TOP 6 PRIORITÁRIOS (Leitura Obrigatória Imediata)
    # -------------------------------------------------------------------------
    ("2606.26487", "01_top6_prioritarios", "01_TempoWAVE_IJCAI2026",
     "Speaking Numbers to LLMs: Multi-Wavelet Number Embeddings for Time Series Forecasting"),
    ("2406.16964", "01_top6_prioritarios", "02_Are_LMs_Useful_TSF_NeurIPS2024",
     "Are Language Models Actually Useful for Time Series Forecasting?"),
    ("2606.08262", "01_top6_prioritarios", "03_CVAformer_IEEE_TCyb2026",
     "Causal Semantic Alignment for LLM-based Time Series Forecasting"),
    ("2403.07815", "01_top6_prioritarios", "04_Chronos_TMLR2024",
     "Chronos: Learning the Language of Time Series"),
    ("2007.01884", "01_top6_prioritarios", "05_LPCMCI_Gerhardus_Runge",
     "High-recall causal discovery for autocorrelated time series with latent confounders"),
    ("2410.21328", "01_top6_prioritarios", "06_Deconfounded_TSF_Gao2026",
     "Deconfounded Time Series Forecasting: A Causal Inference Approach"),

    # -------------------------------------------------------------------------
    # 02. PARTE A - NÍVEIS 1 E 2: TOKENIZAÇÃO E INTERFACE NUMÉRICA DO EMBEDDING
    # -------------------------------------------------------------------------
    ("2310.02989", "02_embeddings_tokenizacao", "xVal_Continuous_Tokenization",
     "xVal: A Continuous Numerical Tokenization for Scientific Language Models"),
    ("2502.09741", "02_embeddings_tokenizacao", "FoNE_Fourier_Number_Embedding",
     "FoNE: Precise Single-Token Number Embeddings via Fourier Features"),
    ("2510.06824", "02_embeddings_tokenizacao", "BitTokens_Single_Token_Numeracy",
     "Efficient numeracy in language models through single-token number embeddings (BitTokens)"),
    ("2511.03570", "02_embeddings_tokenizacao", "TabGemma_Tabular_ICL",
     "TabGemma: Text-Based Tabular ICL via LLM using Continued Pretraining and Retrieval"),
    ("2609.01386", "02_embeddings_tokenizacao", "Tokenization_Secretly_Output_Supervision_EMNLP2026",
     "When Tokenization is Secretly Output Supervision"),
    ("2604.16775", "02_embeddings_tokenizacao", "Representation_Before_Training_Fused_Deciles",
     "Representation Before Training: A Practical Benchmark for Generative Medical Event Model Tokenization"),
    ("2610.05726", "02_embeddings_tokenizacao", "FreSia_Frequency_Semantic_Alignment",
     "FreSia: Frequency-Semantic Instantiation and Alignment for Multivariate Time Series Analysis"),
    ("2606.27731", "02_embeddings_tokenizacao", "SMMD_Metric_Loss_Numeric_Tokens",
     "Soft Maximum Mean Discrepancy Loss for Numeric Token Distance Supervision"),
    ("2402.16412", "02_embeddings_tokenizacao", "TOTEM_Tokenized_TS_Embeddings_VQVAE",
     "TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis"),

    # -------------------------------------------------------------------------
    # 03. PARTE A - NÍVEIS 3 E 4: ENCODERS ACOPLADOS, PROMPTS E TSFMs
    # -------------------------------------------------------------------------
    ("2211.14730", "03_encoders_tsfm", "PatchTST_ICLR2023",
     "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"),
    ("2302.13971", "03_encoders_tsfm", "GPT4TS_One_Fits_All_NeurIPS2023",
     "One Fits All: Power General Time Series Analysis by Pretrained LM"),
    ("2310.01728", "03_encoders_tsfm", "Time_LLM_Reprogramming_ICLR2024",
     "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"),
    ("2310.07820", "03_encoders_tsfm", "LLMTime_ZeroShot_Gruver_NeurIPS2023",
     "Large Language Models Are Zero-Shot Time Series Forecasters (LLMTime - ID Corrigido)"),
    ("2310.10688", "03_encoders_tsfm", "TimesFM_Google_ICML2024",
     "A decoder-only foundation model for time-series forecasting (TimesFM)"),
    ("2402.02368", "03_encoders_tsfm", "Timer_Large_TS_Models_ICML2024",
     "Timer: Generative Pre-trained Transformers Are Large Time Series Models"),
    ("2402.02592", "03_encoders_tsfm", "Moirai_Universal_TS_Transformers_ICML2024",
     "Unified Training of Universal Time Series Forecasting Transformers (Moirai)"),
    ("2403.00131", "03_encoders_tsfm", "UniTS_Unified_MultiTask_NeurIPS2024",
     "UniTS: A Unified Multi-Task Time Series Model"),
    ("2402.10835", "03_encoders_tsfm", "Tang_TSF_with_LLMs_Capabilities",
     "Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities"),
    ("2410.11674", "03_encoders_tsfm", "LLM_Mixer_Multiscale",
     "LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting"),
    ("2502.05699", "03_encoders_tsfm", "Context_More_Important_Than_Reasoning",
     "Context information can be more important than reasoning for time series forecasting with a LLM"),
    ("2506.21570", "03_encoders_tsfm", "Random_Init_Cant_Catch_Up_LM_Transfer",
     "Random Initialization Can't Catch Up: The Advantage of Language Model Transfer for TSF"),
    ("2508.04239", "03_encoders_tsfm", "DP_GPT4MTS_Dual_Prompt",
     "DP-GPT4MTS: Dual-Prompt Large Language Model for Textual-Numerical Time Series"),
    ("2512.20140", "03_encoders_tsfm", "Noise_Injection_ZeroShot_LLM",
     "Enhancing Zero-Shot Time Series Forecasting in Off-the-Shelf LLMs via Noise Injection"),

    # -------------------------------------------------------------------------
    # 04. PARTE B - DESCOBERTA CAUSAL EM SÉRIES TEMPORAIS COM CONFUNDIDORES
    # -------------------------------------------------------------------------
    ("2002.00498", "04_descoberta_causal", "DYNOTEARS_Structure_Learning_AISTATS2020",
     "DYNOTEARS: Structure Learning from Time-Series Data"),
    ("2112.08417", "04_descoberta_causal", "Causal_Ancestral_Graphs_DPAGs_AnnStat",
     "Characterization of causal ancestral graphs for time series with latent confounders"),
    ("2202.11286", "04_descoberta_causal", "Deep_Granger_Latent_Confounding",
     "Deep Recurrent Modelling of Granger Causality with Latent Confounding"),
    ("2209.03427", "04_descoberta_causal", "LPCMCI_Evaluation_Reiser",
     "Causal discovery for time series with latent confounders"),
    ("2209.11497", "04_descoberta_causal", "SCEVAE_Knockoff_Interventions_NeurIPSW",
     "Time Series Causal Link Estimation under Hidden Confounding using Knockoff Interventions"),
    ("2306.12896", "04_descoberta_causal", "J_PCMCI_Plus_Multiple_Datasets",
     "Causal discovery for time series from multiple datasets with latent contexts"),
    ("2406.01161", "04_descoberta_causal", "Dynamic_SCM_Boeken_Mooij",
     "Dynamic Structural Causal Models"),
    ("2607.12140", "04_descoberta_causal", "Dynamic_SCM_Extended_2026",
     "Dynamic Structural Causal Models (Revised and Extended Version)"),
    ("2408.05584", "04_descoberta_causal", "CIC_Dynamical_Causality_Invisible_Confounders",
     "Dynamical causality under invisible confounders"),
    ("2504.20172", "04_descoberta_causal", "Causal_Identification_TS_Models",
     "Causal Identification in Time Series Models"),
    ("2610.05618", "04_descoberta_causal", "ORACLE_VARX_ACLE_Orthogonalized_DML",
     "Causal Lag Structure Discovery in Confounded Time Series via Orthogonalized Adaptive Estimation"),
    ("2610.13430", "04_descoberta_causal", "Guide_TS_Causal_Discovery_Taxonomy2026",
     "Methods and Taxonomy for Time-Series Causal Discovery"),

    # -------------------------------------------------------------------------
    # 05. PARTE B - DECONFOUNDING, AUDITORIA DE RISCO E INVARIÂNCIA EM TSF
    # -------------------------------------------------------------------------
    ("2111.14820", "05_deconfounding_invariancia", "Causal_Motion_Forecasting_CVPR2022",
     "Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective"),
    ("2403.08572", "05_deconfounding_invariancia", "Caformer_Causal_Perspective",
     "Caformer: Rethinking Time Series Analysis from Causal Perspective"),
    ("2403.11960", "05_deconfounding_invariancia", "Casper_Frontdoor_GNN_CIKM2024",
     "Causality-Aware Spatiotemporal Graph Neural Networks for Spatiotemporal TS Imputation"),
    ("2405.05430", "05_deconfounding_invariancia", "Invariant_TSF_Smart_Cities",
     "Towards Invariant Time Series Forecasting in Smart Cities"),
    ("2408.12063", "05_deconfounding_invariancia", "Climate_Deconfounding_Factor_Model_IJCAI2025",
     "Deconfounding Multi-Cause Latent Confounders: A Factor-Model Approach to Climate Bias Correction"),
    ("2604.02488", "05_deconfounding_invariancia", "Causal_Audit_Risk_Assessment",
     "Causal-Audit: A Framework for Risk Assessment of Assumption Violations in TS Causal Discovery"),
    ("2608.00198", "05_deconfounding_invariancia", "AutoCause_Automated_Causal_Discovery",
     "AutoCause: A Python framework that automates expert decisions in environmental TS causal discovery"),
    ("2609.31315", "05_deconfounding_invariancia", "LUCID_Confounding_Regime_Routing",
     "LUCID: Learning Under Confounding for Inference and Discovery in Time Series"),

    # -------------------------------------------------------------------------
    # 06. INTERPRETABILIDADE MECANÍSTICA E BENCHMARKS DE CAUSALIDADE
    # -------------------------------------------------------------------------
    ("2603.10071", "06_benchmarks_interpretabilidade", "Dissecting_Chronos_SAE_ICLR2026W",
     "Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in TSFMs"),
    ("2509.05801", "06_benchmarks_interpretabilidade", "time2time_Causal_Intervention_Hidden_States",
     "time2time: Causal Intervention in Hidden States to Simulate Rare Events in TSFMs"),
    ("2310.01753", "06_benchmarks_interpretabilidade", "CausalTime_Benchmark_ICLR2024",
     "CausalTime: Realistically Generated Time-series for Benchmarking of Causal Discovery"),
    ("2506.01361", "06_benchmarks_interpretabilidade", "TimeGraph_Benchmark_KDD2025",
     "TimeGraph: Synthetic Benchmark Datasets for Robust Time-Series Causal Discovery"),
    ("2609.11897", "06_benchmarks_interpretabilidade", "CausalArena_Benchmark_FM_Era",
     "CausalArena: Benchmarking Causal Discovery in the Foundation Model Era"),
]


def baixar_pdf_arxiv(
    arxiv_id: str,
    destino: Path,
    max_tentativas: int = 4,
    pausa_base: float = 3.5
) -> Tuple[bool, str]:
    """
    Baixa um PDF do arXiv com cabeçalho User-Agent adequado, validação de
    assinatura binária '%PDF-' e exponential backoff para respeitar rate-limits.
    """
    if destino.exists() and destino.stat().st_size > 10_000:
        return True, "Já existente no disco (ignorado)"

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {
        "User-Agent": (
            "AutoTimesResearchBot/1.0 "
            "(Academic Literature Review; mailto:research@autotimes.org)"
        ),
        "Accept": "application/pdf",
    }

    for tentativa in range(1, max_tentativas + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as response:
                conteudo = response.read()

                # Verifica se o arquivo retornado é de fato um PDF válido
                if not conteudo.startswith(b"%PDF-"):
                    raise ValueError(
                        "Resposta recebida não possui cabeçalho %PDF- válido (possível bloqueio ou HTML)."
                    )

                destino.parent.mkdir(parents=True, exist_ok=True)
                with open(destino, "wb") as f:
                    f.write(conteudo)

                tamanho_kb = len(conteudo) / 1024.0
                return True, f"Baixado com sucesso ({tamanho_kb:.1f} KB)"

        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            espera = pausa_base * (2 ** (tentativa - 1))
            print(f"  [Aviso] Tentativa {tentativa}/{max_tentativas} falhou para {arxiv_id}: {exc}")
            if tentativa < max_tentativas:
                print(f"          Aguardando {espera:.1f}s antes de tentar novamente...")
                time.sleep(espera)
            else:
                return False, f"Falha após {max_tentativas} tentativas: {exc}"

    return False, "Erro desconhecido"


def executar_pipeline_download() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    manifesto_path = BASE_DIR / "manifesto_acervo.csv"
    registros_manifesto: List[Dict[str, str]] = []

    total = len(CATALOGO_ARTIGOS)
    sucessos = 0
    falhas = 0

    print("=" * 78)
    print(f"Iniciando download do acervo científico AutoTimes ({total} artigos)")
    print(f"Diretório de destino: {BASE_DIR.resolve()}")
    print("=" * 78)

    for idx, (arxiv_id, subpasta, nome_curto, titulo) in enumerate(CATALOGO_ARTIGOS, start=1):
        nome_arquivo = f"{arxiv_id}_{nome_curto}.pdf"
        caminho_pdf = BASE_DIR / subpasta / nome_arquivo

        print(f"[{idx:02d}/{total:02d}] {arxiv_id} | {nome_curto}")
        print(f"        Título: {titulo}")

        ja_existia = caminho_pdf.exists() and caminho_pdf.stat().st_size > 10_000
        ok, msg = baixar_pdf_arxiv(arxiv_id, caminho_pdf)
        print(f"        Status: {msg}")

        if ok:
            sucessos += 1
        else:
            falhas += 1

        registros_manifesto.append({
            "ordem": str(idx),
            "arxiv_id": arxiv_id,
            "categoria": subpasta,
            "nome_curto": nome_curto,
            "titulo": titulo,
            "url_abs": f"https://arxiv.org/abs/{arxiv_id}",
            "url_pdf": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "caminho_local": str(caminho_pdf),
            "status_download": "OK" if ok else "ERRO",
            "detalhe": msg,
        })

        # Pausa cortês de 3.5s apenas quando houve requisição real de rede
        if not ja_existia and idx < total:
            time.sleep(3.5)

    # Salva o manifesto em CSV para fácil consulta e rastreabilidade
    campos = [
        "ordem", "arxiv_id", "categoria", "nome_curto", "titulo",
        "url_abs", "url_pdf", "caminho_local", "status_download", "detalhe"
    ]
    with open(manifesto_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros_manifesto)

    print("=" * 78)
    print(f"Processo concluído: {sucessos}/{total} disponíveis | {falhas} falhas.")
    print(f"Manifesto salvo em: {manifesto_path.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    executar_pipeline_download()