# Spec Delta

## Purpose

Controla como a série decomposta é escrita no documento de entrada do modelo
(política de data, casas decimais, codificação dos valores e marcadores) e o
custo em tokens/orçamento visível que cada escolha produz, mantendo a escrita
atual como padrão intacto para que os resultados continuem comparáveis com os
vereditos históricos.

## ADDED Requirements

### Requirement: Padrão idêntico ao texto atual
Quando nenhuma variante de texto for informada, o sistema SHALL gerar o
documento exatamente como hoje (mesmos marcadores, mesma data por extenso, 6
casas decimais por extenso, mesmos separadores) e SHALL reaproveitar os caches
existentes sem regerá-los.

#### Scenario: Rodada sem flags novas reusa o cache
- **WHEN** o runner é executado com `--texto-decomp vmd` e sem as novas flags
  de texto
- **THEN** a tag de cache é a mesma das rodadas anteriores e nenhum documento
  é regenerado

#### Scenario: Documento byte a byte igual
- **WHEN** o mesmo janela é escrita com o padrão antes e depois desta mudança
- **THEN** os dois textos são idênticos caractere a caractere

### Requirement: Política de data no Pseudo-JSON
O sistema SHALL permitir escolher como o carimbo temporal aparece no
Pseudo-JSON, com as opções `completa` (mês/dia/hora por extenso, padrão),
`curta`, `hora`, `indice` e `nenhuma`; a opção escolhida SHALL valer para
todos os pontos do documento.

#### Scenario: Remover a data
- **WHEN** a política de data é definida como `nenhuma`
- **THEN** o documento não contém nenhum token de data e cada ponto começa
  direto nos componentes `/t /s /r`

#### Scenario: Data completa é o padrão
- **WHEN** nenhuma política é informada
- **THEN** cada ponto traz mês, dia e hora por extenso, como hoje

#### Scenario: Redução de custo medida
- **WHEN** a política `completa` é trocada por `nenhuma` no mesmo documento
- **THEN** o custo por ponto cai e a quantidade de horas visíveis nos 896
  tokens de orçamento aumenta, sem alterar o orçamento

### Requirement: Casas decimais no Char-by-Char
O sistema SHALL permitir escolher as casas decimais escritas por extenso no
Char-by-Char (6, 4, 3 ou 2; padrão 6) e SHALL arredondar por truncamento
determinístico, sem notação científica.

#### Scenario: Menos decimais cabem mais pontos
- **WHEN** os decimais saem de 6 para 3 na mesma janela
- **THEN** a frase T/S/R fica menor em tokens e o número de pontos visíveis
  dentro das páginas alocadas (3+2+2) aumenta

#### Scenario: Truncamento determinístico
- **WHEN** o mesmo valor é escrito duas vezes com a mesma quantidade de
  decimais
- **THEN** as duas saídas são idênticas e sem notação científica

### Requirement: Codificação dos valores
O sistema SHALL permitir escolher a codificação numérica do Char-by-Char
entre `extenso` (palavras, padrão) e `digitos` (algarismos crus), mantendo a
mesma normalização local e o mesmo contrato `max_abs`.

#### Scenario: Codificação por algarismos
- **WHEN** a codificação é definida como `digitos`
- **THEN** os valores são escritos com algarismos e o custo por ponto cai em
  relação a `extenso`, mantendo normalização e `max_abs` inalterados

### Requirement: Relatório de custo em tokens
O sistema SHALL imprimir, no banner da rodada, quantos tokens cada variante
custa por ponto e quantos pontos/horas ficam visíveis no orçamento de 896
tokens, para a configuração escolhida.

#### Scenario: Banner mostra o custo
- **WHEN** a rodada inicia com variantes de texto selecionadas
- **THEN** o banner lista tok/ponto e pontos visíveis da variante, ao lado do
  valor padrão correspondente

### Requirement: Protocolo de orçamento e paginação intocados
O sistema SHALL manter o orçamento de 7 páginas × 128 tokens, a paginação
pela cabeça do texto e a divisão 3+2+2 entre T/S/R; nenhuma variante de texto
pode alterar modelo, épocas, seeds ou checkpoints.

#### Scenario: Variantes não mudam o orçamento
- **WHEN** qualquer variante de texto é selecionada
- **THEN** o total de tokens retidos segue 896 por janela e a divisão de
  páginas segue 3+2+2

### Requirement: Reprodutibilidade e cache por variante
O sistema SHALL gerar texto determinístico para a mesma entrada + flags e SHALL
incluir as variantes de texto na tag de cache, de modo que duas variantes
diferentes nunca compartilhem o mesmo cache. Cada cache SHALL depender apenas
dos eixos que mudam aquele documento: o Pseudo-JSON só do eixo de data e o
Char-by-Char só das casas decimais e da codificação.

#### Scenario: Duas variantes não colidem
- **WHEN** duas rodadas rodam com decimais diferentes (6 e 3)
- **THEN** cada uma grava seu próprio arquivo de cache e nenhuma lê o cache da
  outra

#### Scenario: Eixo que não muda o documento não regenera cache
- **WHEN** uma rodada seleciona uma política de data diferente do padrão
- **THEN** o cache do Char-by-Char (650 MB) permanece o do padrão e não é
  regenerado; o inverso vale para decimais/codificação em relação ao cache do
  Pseudo-JSON

### Requirement: Reformulação da escrita sem mudar o conteúdo
O sistema SHALL permitir reformular a escrita dos componentes do
Pseudo-JSON — casas decimais de `/t /s /r` (2, 1 ou 0; padrão 2),
marcadores (`normal` com espaços soltos, padrão, ou `compacta`) e escala
(`float`, padrão, ou `inteiro` = valores ×100 sem ponto decimal) — e do
Char-by-Char — arredondamento (`truncar`, padrão histórico, ou `arredondar`
metade para longe do zero) e política de zeros à direita (`manter`, padrão,
ou `remover`) — mantendo em qualquer variante os mesmos valores (dentro da
tolerância do formato), a mesma normalização local, `max_abs` e contrato de
tensores, e mantendo o padrão byte a byte idêntico ao atual.

#### Scenario: Escala inteira preserva os valores
- **WHEN** `pj_escala=inteiro` é selecionada
- **THEN** cada componente aparece como inteiro ×100 (`/t 1519`) e o valor
  recuperado difere do original em menos de 0.005, sem token de ponto

#### Scenario: Arredondar elimina o viés do truncamento
- **WHEN** `arredondamento=arredondar` é aplicado a 0.0005 com 3 casas
- **THEN** o texto traz 0.001 (metade para longe do zero), enquanto
  `truncar` traz 0.000; o custo em tokens é o mesmo e o padrão não muda

#### Scenario: Zeros à direita economizam sem mudar o valor
- **WHEN** `zeros=remover` é aplicado a um ponto com cauda de zeros
- **THEN** a frase termina no último dígito significativo (`menos zero ponto
  cinco`), nenhum ponto termina em `zero` fracionário, e o texto padrão
  (todas as casas escritas) permanece byte a byte idêntico

### Requirement: Execução seletiva de modelo textual
O sistema SHALL permitir rodar apenas um dos modelos textuais da bateria
(`--so-modelo pj|cbc`, padrão `ambos`), pulando a geração e o treino do outro;
o modelo não rodado SHALL ficar fora do JSON do veredito e aparecer como não
rodado nos relatórios, sem alterar o resultado do modelo executado.

#### Scenario: Variante de um só lado roda metade do caminho
- **WHEN** o runner é executado com `--so-modelo cbc` (ex.: uma variante de
  decimais, que só afeta o Char-by-Char)
- **THEN** o Pseudo-JSON não é gerado nem treinado, o JSON traz apenas o
  Char-by-Char, e a tabela de veredito marca a linha ausente como não rodada
