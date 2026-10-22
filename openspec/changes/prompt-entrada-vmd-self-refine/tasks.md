# Tasks

## 1. Base de segurança (padrão byte a byte)

- [x] 1.1 Criar script `testes_letras/smoke_texto_padrao.py` que gera o documento Pseudo-JSON e o Char-by-Char de uma janela VMD fixa e compara com golden salvo em disco; verificação: `python -m testes_letras.smoke_texto_padrao` imprime `OK (identico ao golden)`.
- [x] 1.2 Confirmar que rodada sem flags novas mantém a tag de cache atual; verificação: `python -m testes_letras.run_apples_to_apples_trio --texto-decomp vmd ... --dry-run-cache` (ou checagem manual da tag) mostra o mesmo hash dos `char_by_char_*.npz` / `pseudo_json_*.npz` existentes.

## 2. Encanamento das opções de texto (flags no runner)

- [x] 2.1 Criar o dataclass congelado `FormatoTexto` (data, casas, codificação) e injetá-lo como **um** elemento do `args_tuple` (19 → 20 campos), sem alterar o comportamento padrão; verificação: smoke de integração imprime `20 campos` e o teste dourado do grupo 1 continua `OK`.
- [x] 2.2 Adicionar as flags no CLI de `run_apples_to_apples_trio.py` (`--texto-data`, `--texto-decimais`, `--texto-encoding`), com ajuda mostrando o padrão; verificação: `python -m testes_letras.run_apples_to_apples_trio --help` lista as três flags com `[padrao: ...]`.
- [x] 2.3 Documentar as novas flags nos docstrings de `construir_dataset_char_by_char` e `texto_char_por_char`; verificação: `py_compile` passa e o `--help` continua sem erro de escape (`_pct`).

## 3. Política de data no Pseudo-JSON

- [x] 3.1 Implementar a política de data (`completa`/`curta`/`hora`/`indice`/`nenhuma`) numa única função usada por `_pseudo_json_mstl`, mantendo `completa` idêntica ao formato literal atual; verificação: golden do grupo 1 (PJ) continua `OK`.
- [x] 3.2 Smoke `testes_letras/smoke_pj_data.py` gerando as 5 variantes da mesma janela e medindo tok/ponto com o GPT-2 local; verificação: imprime a tabela (completa ~25,5 / nenhuma ~16,3 tok/ponto) e confere que `nenhuma` não contém token de data.
- [x] 3.3 Incluir a variante de data na tag de cache só quando não-padrão; verificação: duas rodadas (`--texto-data completa` e `--texto-data nenhuma`) geram dois arquivos de cache distintos e a rodada padrão reusa o hash antigo.

## 4. Decimais e codificação no Char-by-Char

- [x] 4.1 Parametrizar `casas` em `_escalar_para_inteiros`/`_montar_texto` (6/4/3/2, padrão 6, truncamento aritmético preservado, sem notação científica); verificação: `python -m testes_letras.smoke_texto_padrao` segue `OK` com o padrão, e o smoke `smoke_cbc_decimais.py` mostra as 4 variantes sem NaN/`e+` no texto.
- [x] 4.2 Implementar a codificação `digitos` (algarismos crus) como opção separada dos decimais, mantendo normalização local e `max_abs`; verificação: smoke imprime tok/ponto menor que `extenso` e o contrato `max_abs` do dataset permanece idêntico ao da variante `extenso`.
- [x] 4.3 Incluir decimais e codificação na tag de cache quando não-padrão; verificação: rodadas 6×`extenso` vs 3×`extenso` gravam caches separados e a padrão reusa o hash atual.
- [x] 4.4 Atualizar a documentação das flags e a ajuda do CLI mostrando os padrões; verificação: `--help` exibe `[padrao: 6]` e `[padrao: extenso]`.

## 5. Relatório de custo em tokens no banner

- [x] 5.1 Calcular tok/ponto e pontos visíveis no orçamento 896 sobre UMA janela real com o tokenizer já carregado e imprimir no banner, ao lado do valor padrão; verificação: rodada de smoke mostra no banner `tok/ponto` e `pontos visiveis` da variante escolhida junto do `[padrao: ...]`, com números na mesma ordem de grandeza do medido (~16,3 PJ / ~16,7 CbC).

## 6. Pacote `testes_letras/self_refine/`

- [x] 6.1 Criar `self_refine/validadores.py` com as checagens puras (contrato das 3 frases T/S/R, finitude, reconstrução T+S+R na tolerância, contrato de tensores ids/lens/max_abs, orçamento 7 páginas / 896 tokens) e códigos de falha + regra de correção por código; verificação: `python -m testes_letras.smoke_self_refine` passa os 5 cenários (doc válido e 1 falha por verificador).
- [x] 6.2 Criar `self_refine/memoria.py` (leitura tolerante a ausente/corrompido, escrita atômica `tmp`+`os.replace`, FIFO com teto de entradas) em `.runtime/autotimes_memory.json`; verificação: smoke grava → relê → apaga/corrompe → relê sem exceção e com aviso; `git status` não lista o arquivo.
- [x] 6.3 Criar `self_refine/agente.py` com o template de prompt de sistema em português usando as tags `<context_analysis>`, `<initial_draft>`, `<evaluation>`, `<self_reflection_and_feedback>`, `<refined_output>`, `<final_result>`, com os campos de contexto (dados, regras, memória episódica) preenchíveis; verificação: smoke imprime o prompt renderizado e confere que as 6 tags e os 3 campos aparecem.
- [x] 6.4 Criar `self_refine/loop.py` (tentativa → validação → feedback pela regra → retry com limite, registro na memória só no caminho de falha, aborto explícito ao esgotar); verificação: smoke simula falha determinística → 2ª tentativa corrigida → memória grava `{motivo, regra, contexto}` → com limite=1 aborta com a última falha na mensagem.

## 7. Integração da flag `--self-refine`

- [x] 7.1 Adicionar `--self-refine` (padrão off), `--self-refine-max-tentativas` (3) e `--self-refine-memoria` no CLI, com ajuda dizendo claramente que é uma flag opcional; verificação: `--help` mostra as flags com `[padrao: off]`.
- [x] 7.2 Ligar o loop à geração de documentos (PJ e CbC) dentro de `construir_dataset_*`, com validação **antes** de persistir em cache/treino, feita só no processo pai (workers devolvem registros de falha); verificação: smoke com flag ligada e com falha injetada mostra o documento recusado, o retry e o cache não gravado.
- [x] 7.3 Banner/estado da flag: imprimir `SELF-REFINE: ATIVO` quando ligada e **nada** quando desligada; verificação: duas rodadas de smoke — com a flag o banner tem a linha, sem a flag o log não menciona Self-Refine.
- [x] 7.4 Adicionar `.runtime/` ao `.gitignore`; verificação: `git check-ignore .runtime/autotimes_memory.json` retorna o caminho.
- [x] 7.5 Smoke de regressão: rodar o mesmo smoke do grupo 1 **sem** a flag e confirmar idêntico ao golden (comportamento atual intocado); verificação: `python -m testes_letras.smoke_texto_padrao` imprime `OK (identico ao golden)`.

## 8. Integração e verificação final

- [x] 8.1 `py_compile` em todos os arquivos alterados + `--help` completo do runner sem erro; verificação: compilação sem saída de erro e help exibido.
- [x] 8.2 Smoke de integração de dataset com as duas escritas e as variantes ligadas (`19→20 campos`, contrato 896/7/(3,3,1)); verificação: script imprime `INTEGRACAO OK` e o contrato bate.
- [x] 8.3 Rodada rápida de 1 época (só semeadura) com e sem `--self-refine` para confirmar que o fluxo de treino não muda; verificação: as duas completam e os logs mostram o mesmo número de janelas/tokens retidos.

## 9. Agilidade dos experimentos A/B de texto (pós-8.3)

- [x] 9.1 Desacoplar a tag de cache por eixo: PJ só leva `data` (`tag_cache_pj`), CbC só leva `casas`/`encoding` (`tag_cache_cb`); o padrão continua sem sufixo e reusa os caches históricos; verificação: `python -m testes_letras.smoke_cache_tag` OK, cobrindo o desacoplamento (variante de data não regenera o cache do CbC e vice-versa).
- [x] 9.2 Flag `--so-modelo pj|cbc|ambos` (padrão `ambos`): pula geração e treino do modelo textual não afetado; o modelo ausente fica fora do JSON e aparece como "(nao rodado)" na tabela, e os gráficos só plotam as séries presentes; nome do JSON ganha o sufixo da variante de texto e do `so-modelo`; verificação: `python -m testes_letras.smoke_so_modelo` OK e golden do grupo 1 continua `OK (identico ao golden)`.
- [x] 9.3 Sessão A/B de texto (10 épocas × seed 2021, no terminal do usuário): as 8 rodadas completaram (logs `ab_*.txt`, vereditos `veredito_trio_txt_vmd_txt*.json`); verificação: comparação contra o baseline seed 2021 (PJ 0.359157 / CbC 0.357060) — PJ `nenhuma` **0.356939** (−0.0022, única que melhora; `curta` 0.357973, `indice` 0.360654, `hora` 0.361818); CbC todas piores (`digitos` 0.358992, `dec2` 0.359279, `dec4` 0.360516, `dec3` 0.360845). Conclusão: só remover a data do PJ ajuda; redução de casas/encoding no CbC piora — nenhuma variante da sessão vira padrão.

## 10. Eixos de reformulação do prompt (A+B+C no PJ, arredondamento/zeros no CbC)

- [x] 10.1 Eixos do PJ no `FormatoTexto`/CLI: `--texto-pj-casas` (2|1|0), `--texto-pj-marcacao` (normal|compacta), `--texto-pj-escala` (float|inteiro, com validação de incompatibilidade inteiro×casas); verificação: novo `smoke_pj_escrita` OK (fidelidade dos valores parseados, determinismo, sem notação científica, padrao byte a byte) + golden do grupo 1 continua `OK (identico ao golden)`.
- [x] 10.2 Eixos do CbC: `--texto-arredondamento` (truncar|arredondar — metade para longe do zero por aritmética de inteiros) e `--texto-zeros` (manter|remover — corta o `zero` final); verificação: `smoke_cbc_decimais` OK (casas sintéticas arredondar=[1,-1,0,2] vs truncar=[0,0,0,1], sem `zero` final nos pontos, max_abs inalterado, −0.8% tokens).
- [x] 10.3 Tags de cache por documento para os eixos novos (pj_* só no cache do PJ; arredondamento/zeros só no do CbC), `rotulo()`/`fora_do_padrao()`/banner e JSON atualizados; verificação: `smoke_cache_tag` OK (8 eixos desacoplados) e `smoke_banner_formato` OK.
- [x] 10.4 Triagem das variantes escolhidas (fase 1: 10 épocas × seed 2021; fase 2: confirmação em 3 seeds) — **CONCLUÍDA; decisão: NENHUMA variante vira padrão.** Fase 1: PJ `nenhuma` 0.356939 (replicou o A/B), `pc0` 0.358246, `inteiro` 0.361178, CbC `arredondar` 0.357764. Fase 2 (3 seeds): PJ `nenhuma` **0.3638 ± 0.0050** (2021=0.356939 ep1 / 42=0.368636 ep2 / 1234=0.365946 ep3) **perde para o baseline 0.362145** — o −0,0022 da seed 2021 era sorte de seed (mesmo erro do K=5 de 08/10: vitória single-seed não confirma). Combo CbC `arredondar+remover` 0.3579 (empate com 0,357060). Conclusão: com 3 seeds nenhuma escrita vence o padrão; a variância entre seeds (±0,005) engole qualquer ganho de escrita (≤0,002) — prompt-only está esgotado como caminho ao 0,3560.

## Workflow follow-up

- Sessão A/B (grupo 9.3) executada no terminal do usuário e concluída: 8/8 rodadas, logs em `testes_letras/recente/logs/ab_*.txt`.
- Rodar a fila 10.4 fase 1 (5 rodadas, 10 épocas × só a seed 2021, ~5h20, script `queue_104.ps1` em Temp\opencode) no terminal do usuário; comparar contra o baseline seed 2021 (PJ 0.359157 / CbC 0.357060) e confirmar os vencedores em 3 seeds antes de decidir padrão; referências gerais: vereditos atuais (PJ 0.3621 / CbC 0.3606), textuais (CEEMDAN 0.3634/0.3633, MSTL 0.3609/0.3612) e o numérico 0.3539.
- Definir o escopo de commit (o working tree já carrega ~1500 linhas pré-existentes nos mesmos arquivos; nunca adicionar `CEEMDAN-GPU/`).
- Archive the change after the project's review requirements are satisfied.
