# Spec Delta

## Purpose

Ciclo opcional de Self-Refine com memória episódica aplicado à geração dos
documentos textuais: valida programaticamente o que foi escrito antes de entrar
no cache/treino, refaz com o feedback da falha e persiste a lição entre
execuções — tudo local, sem API externa, e desligado por padrão.

## ADDED Requirements

### Requirement: Self-Refine é uma flag, desligada por padrão
O sistema SHALL expor o Self-Refine como uma flag de CLI (padrão: desligada).
Com a flag ausente o comportamento SHALL ser idêntico ao atual: nenhuma
memória episódica, nenhuma validação adicional e nenhum retry serão executados
nem reportados.

#### Scenario: Rodada normal sem a flag
- **WHEN** o runner é executado sem `--self-refine`
- **THEN** nenhuma checagem, memória ou retry acontece e o log não menciona
  Self-Refine

#### Scenario: Rodada com Self-Refine ligado
- **WHEN** o runner é executado com `--self-refine`
- **THEN** o banner e o log indicam explicitamente que o modo Self-Refine está
  ATIVO, e a validação/retorno acontece para os documentos gerados

### Requirement: Validação programática antes do treino
Com a flag ligada, o sistema SHALL validar cada documento/texto gerado antes de
persistir em cache ou entrar no treino: contrato das três frases (T/S/R) com
marcadores, finitude dos valores, reconstrução da decomposição (T+S+R igual à
série de origem dentro da tolerância), contrato dos tensores (ids, lens,
max_abs) e orçamento de 7 páginas / 896 tokens.

#### Scenario: Documento inválido é detectado
- **WHEN** um documento gerado tem uma frase ausente, valor NaN ou reconstrução
  fora da tolerância
- **THEN** o documento NÃO é salvo em cache nem treinado e a falha é registrada

#### Scenario: Documento válido segue fluxo normal
- **WHEN** todas as checagens passam
- **THEN** o documento é cacheado e segue para o treino sem alteração

### Requirement: Memória episódica persistida entre execuções
Com a flag ligada, o sistema SHALL gravar em `.runtime/autotimes_memory.json`
o motivo de cada falha e a regra de correção aplicada, junto do contexto que a
produziu (decomposição, variante de texto, feature/split), e SHALL recarregar
esse histórico no início de cada nova execução.

#### Scenario: Falha vira lição
- **WHEN** uma validação falha e o retry corrige o problema
- **THEN** motivo, regra de correção e contexto são gravados no arquivo de
  memória

#### Scenario: Próxima execução lê a memória
- **WHEN** uma nova rodada inicia com a flag ligada e já existe histórico
- **THEN** o histórico é carregado e disponibilizado ao fluxo de geração antes
  da primeira tentativa

#### Scenario: Estado ausente ou corrompido não derruba a rodada
- **WHEN** o arquivo de memória não existe, está vazio ou é inválido
- **THEN** o sistema começa com memória vazia e emite um aviso, sem interromper

### Requirement: Retry com feedback e limite de tentativas
Com a flag ligada, quando a validação falhar o sistema SHALL gerar nova
tentativa injetando o motivo da falha e a regra de correção como feedback,
até um número limitado de tentativas; esgotado o limite, a execução SHALL
falhar de forma explícita em vez de usar o documento inválido.

#### Scenario: Retry corrige com base no feedback
- **WHEN** a primeira tentativa falha na validação
- **THEN** a segunda tentativa recebe o feedback da falha e o motivo é
  registrado na memória

#### Scenario: Limite esgotado
- **WHEN** todas as tentativas permitidas falham
- **THEN** a rodada aborta com mensagem clara indicando a última falha, sem
  gravar cache

### Requirement: Operação 100% local, sem dependência nova
O sistema SHALL operar sem nenhuma chamada de API de LLM externa e sem novas
dependências de biblioteca; toda validação e todo refinamento usam apenas
regras do próprio pipeline e componentes já presentes no repositório.

#### Scenario: Sem chave de API
- **WHEN** a flag é ligada em máquina sem qualquer chave de API configurada
- **THEN** o Self-Refine funciona normalmente

### Requirement: Estado não versionado pelo Git
O arquivo de memória SHALL ficar fora do controle de versão, com entrada no
`.gitignore`, e sua escrita SHALL ser atômica de forma a não correr o arquivo
em caso de interrupção.

#### Scenario: Memória ignorada pelo Git
- **WHEN** `git status` é executado após a primeira escrita da memória
- **THEN** `.runtime/autotimes_memory.json` não aparece como pendente

#### Scenario: Interrupção durante a escrita
- **WHEN** o processo é interrompido no meio de uma gravação
- **THEN** o arquivo anterior permanece legível
