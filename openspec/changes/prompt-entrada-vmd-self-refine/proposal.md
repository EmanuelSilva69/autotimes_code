# Proposal

## Why

A melhor configuração textual hoje (VMD K=7, α=2000, init=0, MA auto) chegou a
**0.3606 ± 0.0025 (Char-by-Char)** e **0.3621 ± 0.0045 (Pseudo-JSON)** contra
0.3539 do numérico, mas o modelo só enxerga uma fração minúscula do documento:
o orçamento é fixo em 7×128 = 896 tokens e a paginação entrega só o **começo**
do texto. Medição com o GPT-2 do repo mostra que a escrita atual desperdiça
orçamento: no Pseudo-JSON cada ponto custa ~25,5 tokens e **13 deles são a data
por extenso** (mês/dia/hora em espanhol, mês repetido 720× por mês); no
Char-by-Char são ~16,7 tokens/ponto por causa das 6 casas decimais por extenso.
Reformular o texto (remover/reduzir data, menos decimais, codificação mais
compacta) **duplica a cobertura visível sem tocar em modelo, arquitetura,
paginação ou épocas** — e portanto sem quebrar a comparação com os vereditos
históricos.

Em paralelo, o pipeline gera documentos textuais sem qualquer verificação do
que produz (contrato das 3 frases, finitude, reconstrução T+S+R, orçamento de
tokens) e não guarda lição nenhuma de falha: o mesmo erro pode entrar no cache
e ser treinado. Um ciclo de **Self-Refine com memória episódica** fecha isso.

## What Changes

- Novo módulo de **formatação do texto de entrada** com variantes selecionáveis
  por flag, mantendo a configuração atual como padrão byte a byte:
  - Pseudo-JSON: política de data (`completa` [atual] | `curta` | `hora` |
    `indice` | `nenhuma`).
  - Char-by-Char: número de decimais (`6` [atual] | `4` | `3` | `2`) e
    codificação dos valores (`extenso` [atual] | `digitos`).
  - Relatório de **custo em tokens** por variante (tok/ponto e pontos visíveis
    no orçamento de 896) impresso no banner da rodada.
- **Self-Refine opcional, ativado por flag** (`--self-refine`, *default
  off*): com a flag desligada o comportamento é exatamente o de hoje (nada de
  memória, validação extra ou retry). Ligada, passa a valer:
  - memória episódica persistida em `.runtime/autotimes_memory.json`
    (**não versionada**, nova entrada no `.gitignore`);
  - validação programática do documento gerado (contrato das frases, tensors,
    finitude, reconstrução, orçamento) **antes** de entrar no cache/treino;
  - retry com o feedback da falha injetado na nova tentativa, com limite de
    tentativas e registro do motivo + regra de correção.
- **Nenhuma chamada de API de LLM**: validação e refinamento são 100% locais
  (regras do próprio pipeline + GPT-2 local já carregado). Zero dependência
  nova, roda offline como o resto do projeto.

## Capabilities

### New Capabilities

- `textual-input-format`: como a série decomposta é escrita no documento de
  entrada do modelo — política de data, casas decimais, codificação dos
  valores, marcadores e o custo em tokens/orçamento visível que cada escolha
  produz; define que a configuração atual é o padrão e que variantes mudam só
  o texto, nunca o modelo, a paginação ou as épocas.
- `self-refine-textual-generation`: ciclo opcional (ligado por flag) de
  geração → validação programática → retry com feedback sobre os documentos
  textuais, com memória episódica persistida entre execuções; define também o
  modo desligado (comportamento idêntico ao atual).

### Modified Capabilities

*(não há capacidades existentes em `openspec/specs/` — projeto ainda sem
specs; portanto nada é modificado, tudo é novo.)*

## Impact

- **Código**: `testes_letras/texto_char_by_char.py` (montagem do texto,
  decimais/encoding), `testes_letras/texto_puro/estruturado.py`
  (`_pseudo_json_mstl` / política de data), `testes_letras/run_apples_to_apples_trio.py`
  (novas flags + banner + tag de cache), novo pacote
  `testes_letras/self_refine/` (memória, validadores, loop), `.gitignore`.
- **Caches**: tag de cache passa a incluir as variantes de texto — rodadas com
  flags no padrão continuam reutilizando os caches atuais.
- **Comparabilidade**: nada muda em `paginacao.py` (cabeça, 7 páginas,
  3+2+2), nem em épocas/seeds/checkpoints; resultados continuam comparáveis
  com os vereditos históricos (numérico 0.3539, MSTL 0.3609, CEEMDAN 0.3634).
- **Dependências**: nenhuma nova (sem LiteLLM/OpenAI/Anthropic).
- **Estado**: `.runtime/autotimes_memory.json` criado no primeiro uso,
  ignorado pelo Git.
