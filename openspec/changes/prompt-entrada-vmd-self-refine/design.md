# Design

## Context

Ver `proposal.md` (Why) para a motivação. Fatos que moldam esta solução:

- Protocolo fixo: orçamento 7×128 = 896 tokens, paginação pela **cabeça**
  (`texto_puro/paginacao.py:135` `janela_deslizante` e `:216` `para_modelo`
  com `paginas[:n_paginas_max]`), divisão 3+2+2, 10 épocas, seeds
  `2021,42,1234`. Documentado em `benchmark_definitivo_textual.py:22`.
- Duas escritas, dois pontos de intervenção:
  - Pseudo-JSON: `texto_puro/estruturado.py::_pseudo_json_mstl`
    (`/time {mes dia hora por extenso} /t %.2f /s %.2f /r %.2f`).
  - Char-by-Char: `texto_char_by_char.py::_montar_texto` +
    `_escalar_para_inteiros` (constante `_MAX_DECIMAIS = 6`, truncamento
    aritmético), marcadores `Tendencia:/Sazonalidade:/Residuos:`.
- Medição com o GPT-2 do repo (`gpt2/`) nesta sessão: Pseudo-JSON ~25,5
  tok/ponto hoje → 36 h visíveis nos 896 tokens; com `nenhuma` ~16,3 → 55 h
  (+53%). Char-by-Char ~16,7 tok/ponto com 6 decimais → 11,0 com 3 → pontos
  visíveis 53 → 81 (+53%).
- O `decode` é **só de escala** (`texto_char_by_char.py:1010`, `--decode`
  desligado por padrão): não existe parser de texto → variante de escrita não
  quebra nenhum inversor.
- Workers loky (`Parallel` em `texto_char_by_char.py`) recebem `args_tuple`
  por pickle — qualquer opção nova tem de ser serializável.

## Goals / Non-Goals

**Goals:**

- Variantes de escrita selecionáveis por flag, com o padrão atual intacto
  byte a byte (cache atual reutilizável).
- Orçamento, paginação, modelo, épocas e seeds intocados — comparação com os
  vereditos históricos preservada.
- Self-Refine **opcional por flag**, desligado por padrão, local (sem API),
  com memória episódica, validação programática e retry com feedback.
- Custo em tokens reportado por rodada para decidir variantes com número.

**Non-Goals:**

- Alterar paginação/orçamento (cabeça, 7 páginas) — decisão do usuário.
- Chamadas a LLM externo (OpenAI/Anthropic/LiteLLM) — deixado como costura
  futura, não implementado.
- Retreinar ou mudar arquitetura/decoder — vem depois, conforme pedido.
- Aplicar Self-Refine a algo além da geração/validação de texto.

## Decisions

**D1 — Opções de texto em um único objeto imutável, não mais kwargs.**
Um dataclass congelado `FormatoTexto` (data de data, casas, codificação) vira
**um** elemento do `args_tuple` (19 → 20 campos), em vez de 4 kwargs novos.
*Alternativa descartada*: espalhar kwargs (o tuple já tem 19 campos e cada
novo parâmetro mexe em 4 lugares — assinatura, unpack, docstring, call).

**D2 — Padrão = constantes atuais, com teste dourado.** As variantes viram
parâmetros cujos valores-padrão são exatamente os atuais (`casas=6`,
política `completa`, codificação `extenso`), e um teste compara o documento
gerado antes/depois caractere a caractere. *Alternativa descartada*:
pós-processar o texto para "desfazer" variantes (frágil, mascararia erro).

**D3 — Política de data numa única função de escolha.** `_tempo_keywords` /
`_pseudo_json_mstl` ganham a política como parâmetro (`completa` mantém o
formato literal atual); `nenhuma`/`hora`/`indice`/`curta` são ramos dessa
função. Uma função só evita que PJ e outros templates divirjam.

**D4 — Decimais como parâmetro do caminho existente.** `_escalar_para_inteiros`
e `_montar_texto` recebem `casas`; a aritmética de truncamento (com o epsilon
direcional) é preservada, `_MAX_DECIMAIS` continua sendo o teto de 6. Só se
mudam os expoentes. *Alternativa descartada*: função separada por casas
(duplicaria a lógica de sinal/ponto/`sep`).

**D5 — Codificação `digitos` opcional, sem risco de inversor.** Confirmado que
não há parser de texto no caminho de saída; escrever algarismos crus só muda a
entrada. Vai no mesmo `FormatoTexto` (separada dos decimais para A/B limpo).

**D6 — Relatório de custo no banner, medido de verdade.** Banner calcula
tok/ponto e pontos visíveis usando o tokenizer já carregado sobre UMA janela
real (não extrapolado), ao lado do valor padrão. Sem custo por rodada.

**D7 — Tag de cache inclui variante só quando não-padrão.** Assim o default
mantém os hashes atuais (`char_by_char_*.npz` / `pseudo_json_*.npz`) e cada
variante ganha arquivo próprio. *Alternativa descartada*: sempre incluir
(quebraria a reutilização de cache do padrão, que é justamente o ponto).

**D8 — Pacote novo `testes_letras/self_refine/` com 4 papéis:**
`memoria.py` (leitura/escrita atômica da memória episódica),
`validadores.py` (funções puras: contrato das frases, finitude, reconstrução
T+S+R, contrato de tensores, orçamento), `agente.py` (renderiza o prompt de
sistema com as tags `<context_analysis>`, `<initial_draft>`, `<evaluation>`,
`<self_reflection_and_feedback>`, `<refined_output>`, `<final_result>`) e
`loop.py` (tentativa → validação → feedback → retry).

**D9 — Sem API: o prompt com tags é o *registro de refinamento*.** Como a
decisão é só-local, `agente.montar_prompt(...)` produz o texto com as tags e o
consumidor é o próprio loop: o campo `<self_reflection_and_feedback>` vira o
feedback injetado na nova tentativa e a entrada gravada na memória;
`<final_result>` é o candidato validado. A função fica como **costura** para um
backend de API no futuro, sem implementar chamada nenhuma agora.
*Alternativa descartada*: chamar um LLM externo (o usuário escolheu só-local;
adicionaria dependência e chave de API).

**D10 — Flag `--self-refine` (padrão off) + `--self-refine-max-tentativas`
(3) + `--self-refine-memoria` (`.runtime/autotimes_memory.json`).** O banner
imprime o estado em caixa alta (`SELF-REFINE: ATIVO` / ausente = nada
impresso), atendendo ao pedido de deixar claro que é uma flag que se aciona.

**D11 — Escrita da memória só no processo pai.** Workers loky devolvem
registros de falha; quem grava é o orquestrador (evita lock/corromper o JSON
com escritas concorrentes) e só grava no caminho de falha — acerto não mexe no
texto (determinismo/cache preservados).

**D12 — Retry por regra de correção, não por "sorte".** Cada validador expõe
um código de falha e a regra associada (ex.: NaN → sanea e re-normaliza;
reconstrução fora da tolerância → refaz a decomposição; orçamento estourado →
trunca na cabeça). O feedback injetado é essa regra + o motivo. Tentativas
limitadas; esgotou → aborta explícito, sem gravar cache.

## Risks / Trade-offs

- [Variantes de texto pioram o veredito] → flags por A/B, um de cada vez, com
  defaults intactos; nenhuma variante vira padrão sem rodada comparável.
- [Inflar caches com muitas variantes] → tag por variante; caches são
  `*.npz` já ignorados pelo Git e regeráveis.
- [Retry duplica tempo de cache quando há falha] → flag off por padrão, limite
  de tentativas, validadores baratos (O(L)).
- [Memória contaminar o texto bem-sucedido e quebrar determinismo] → D11:
  memória só no caminho de falha, nunca no caminho normal.
- [Escritas concorrentes corromperem `.runtime/autotimes_memory.json`] →
  escrita atômica (`tmp` + `os.replace`) feita só pelo pai.
- [`digitos`/menos decimais mudarem a distribuição de tokens e invalidarem
  comparação com rodadas antigas] → D7 (cache separado) e D10 (tudo por flag).
- [Regra de correção errada entrar em loop] → limite de tentativas + registro
  da última falha na mensagem de aborto.

## Migration Plan

Sem migração de dados: tudo atrás de flag desligada por padrão.

1. Land dos testes dourados (padrão byte a byte) antes de qualquer variante.
2. Variantes de texto + banner de custo (flags), mantendo padrão idêntico.
3. Pacote `self_refine` atrás de `--self-refine`.
4. `.gitignore` ganha `.runtime/` (estado não versionado).

Rollback: não passar as flags (comportamento idêntico ao atual); remoção do
código novo não afeta caches existentes.

## Open Questions

- Qual variante de A/B rodar primeiro (remover data no PJ vs menos decimais no
  CbC) — não muda specs nem estrutura das tarefas; ambas entram como tarefas e
  a ordem de execução fica com o usuário na hora de rodar.
