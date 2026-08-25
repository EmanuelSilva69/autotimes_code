"""
amostras - Documentos longos de exemplo em PT-BR e versoes equivalentes EN.

Cada par (PT, EN) representa o MESMO conteudo jornalistico/descritivo sobre
energia eletrica (dominio do ETTh1), com ~200-260 palavras: texto longo o
suficiente para exigir MUITAS paginas de 128 tokens no GPT-2.

Uso:
    from testes_letras.texto_puro import amostras
    pares = amostras.carregar_pares()          # built-ins
    pares = amostras.carregar_pares("pt.txt", "en.txt")  # ou de arquivo
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

AMOSTRAS_PT: List[str] = [
    # 1) Carga eletrica no horario de ponta
    (
        "O Operador Nacional do Sistema Elétrico divulgou nesta terça-feira um novo "
        "balanço sobre a carga de energia elétrica do país, indicando que o consumo médio "
        "registrado durante o horário de ponta aumentou significativamente em relação ao mesmo "
        "período do ano passado. Segundo os técnicos responsáveis pela análise, o crescimento "
        "foi puxado principalmente pelo uso intenso de aparelhos de ar-condicionado nas regiões "
        "mais quentes, combinado com a retomada gradual da atividade industrial após o período "
        "de restrições. A projeção para as próximas semanas aponta que a demanda máxima pode "
        "superar os picos históricos registrados no verão anterior, o que exige planejamento "
        "cuidadoso por parte das distribuidoras. Especialistas recomendam que consumidores "
        "industriais desloquem parte de seus processos para as horas de menor carga, quando a "
        "tarifa é mais baixa e o sistema opera com maior folga. O balanço completo será "
        "apresentado em audiência pública na próxima quinta-feira, com a participação de "
        "representantes de todas as concessionárias do sistema interligado."
    ),
    # 2) Reservatorios das hidreletricas
    (
        "Os reservatórios das hidrelétricas das regiões Sudeste e Centro-Oeste fecharam o mês "
        "com nível médio de armazenamento próximo de sessenta e oito por cento, de acordo com "
        "o último boletim divulgado pelo centro de monitoramento. O volume representa uma "
        "melhora expressiva em comparação com o mesmo período do ano passado, quando o nível "
        "ficou abaixo de quarenta por cento devido à estiagem prolongada. As chuvas acima da "
        "média registradas nas cabeceiras dos principais rios formadores das bacias foram "
        "determinantes para a recuperação dos estoques. Apesar do cenário favorável, os "
        "especialistas alertam que o período seco ainda está longe do fim e recomendam a "
        "manutenção das metas de eficiência no consumo. O comitê de acompanhamento também "
        "avalia a possibilidade de reduzir a geração térmica emergencial nas próximas semanas, "
        "o que pode aliviar os custos repassados aos consumidores na tarifa. As decisões serão "
        "tomadas com base na evolução das afluências e na previsão meteorológica para o próximo "
        "trimestre."
    ),
    # 3) Leilao de energia e preco no mercado spot
    (
        "O leilão de energia nova realizado nesta semana negociou aproximadamente dois mil "
        "megawatts de capacidade, distribuídos entre projetos eólicos, solares e pequenas "
        "centrais hidrelétricas. O preço médio do contrato ficou abaixo do teto estabelecido "
        "pela agência reguladora, sinalizando que a competição entre os geradores segue "
        "acirrada. No mercado de curto prazo, o preço da energia no spot registrou queda "
        "acentuada nos últimos dias, acompanhando a melhora das condições hidrológicas e a "
        "redução da demanda em razão do feriado prolongado. Os analistas de mercado avaliam "
        "que o movimento pode se manter nas próximas semanas caso as previsões de chuva se "
        "confirmem nas regiões de maior peso na geração. As distribuidoras, por sua vez, "
        "monitoram o impacto da variação dos preços na tarifa final paga pelos consumidores "
        "residenciais e comerciais, já que parte dos custos de compra de energia é repassada "
        "automaticamente."
    ),
    # 4) Fontes renovaveis e geracao distribuida
    (
        "A geração distribuída de energia solar fotovoltaica continua crescendo em ritmo "
        "acelerado, com mais de trezentos mil novos sistemas conectados à rede no último "
        "trimestre, segundo o levantamento divulgado pela associação do setor. O crescimento "
        "foi impulsionado pela queda dos preços dos módulos fotovoltaicos e pelo aumento das "
        "tarifas de energia convencional, que tornam o investimento mais atrativo para "
        "residências e pequenos comércios. O prazo médio de retorno do investimento, que era "
        "de cerca de seis anos no início da década, hoje está próximo de três anos e meio na "
        "maioria das regiões. Os especialistas alertam, no entanto, para a necessidade de "
        "reforço das redes de distribuição nas áreas de maior concentração de sistemas, "
        "evitando sobrecargas nos horários de alta geração. As concessionárias vêm modernizando "
        "seus equipamentos e revisando os procedimentos de conexão para acomodar o novo perfil "
        "de consumo e geração."
    ),
    # 5) Manutencao programada de linhas de transmissao
    (
        "A concessionária responsável pela operação das linhas de transmissão da região sul "
        "informou que dará início na próxima segunda-feira à manutenção programada de um dos "
        "principais corredores de escoamento de energia, com duração estimada de dez dias. "
        "Durante o período, a capacidade de transmissão do trecho será reduzida, e parte da "
        "energia será redirecionada por rotas alternativas para garantir o fornecimento às "
        "cidades atendidas. Os serviços incluem a substituição de isoladores, a inspeção "
        "termográfica das torres e a limpeza da faixa de servidão, que apresentou avanço da "
        "vegetação após o período de chuvas. A empresa ressalta que as obras foram planejadas "
        "para coincidir com o período de menor demanda e que não há previsão de interrupção "
        "do fornecimento. Representantes do órgão regulador acompanharão a execução dos "
        "trabalhos e poderão aplicar sanções em caso de descumprimento do cronograma."
    ),
    # 6) Previsao meteorologica e consumo de energia
    (
        "A chegada de uma nova frente fria ao centro-sul do país deve provocar queda acentuada "
        "das temperaturas nos próximos dias, com impacto direto no consumo de energia elétrica. "
        "De acordo com o instituto de meteorologia, a massa de ar polar avança a partir de "
        "quinta-feira, derrubando os termômetros em pelo menos dez capitais e elevando o uso "
        "de aquecedores nas residências. O aumento da demanda no período noturno, coincidindo "
        "com o horário de ponta, deve elevar a carga do sistema elétrico em aproximadamente "
        "quatro por cento. Os centros de despacho já acionaram as usinas térmicas mais "
        "flexíveis como medida preventiva e acompanham o comportamento da carga em tempo "
        "real. Meteorologistas afirmam que o episódio deve durar cerca de uma semana, seguido "
        "de gradual elevação das temperaturas. As distribuidoras recomendam atenção ao uso de "
        "equipamentos de aquecimento simultâneo em horários de maior demanda."
    ),
]

AMOSTRAS_EN: List[str] = [
    (
        "The national power system operator released a new report on Tuesday showing that "
        "average electricity consumption during peak hours increased significantly compared "
        "with the same period last year. According to the analysts responsible for the study, "
        "the growth was driven mainly by the heavy use of air conditioning units in the "
        "warmer regions, combined with the gradual recovery of industrial activity after the "
        "period of restrictions. Projections for the coming weeks suggest that peak demand "
        "may exceed the historical highs recorded last summer, which requires careful "
        "planning on the part of the distribution utilities. Experts recommend that "
        "industrial consumers shift part of their processes to off-peak hours, when tariffs "
        "are lower and the system operates with greater spare capacity. The full report will "
        "be presented at a public hearing next Thursday with representatives of all "
        "concessionaires of the interconnected system."
    ),
    (
        "The reservoirs of the hydroelectric plants in the Southeast and Midwest regions "
        "closed the month with an average storage level close to sixty eight percent, "
        "according to the latest bulletin released by the monitoring center. The volume "
        "represents a significant improvement compared with the same period last year, when "
        "the level remained below forty percent because of the prolonged drought. Rainfall "
        "above the historical average recorded at the headwaters of the main rivers that "
        "form the basins was decisive for the recovery of the stocks. Despite the favorable "
        "scenario, experts warn that the dry season is still far from over and recommend "
        "maintaining the consumption efficiency targets. The monitoring committee is also "
        "evaluating the possibility of reducing emergency thermal generation in the coming "
        "weeks, which could ease the costs passed on to consumers in the tariff. Decisions "
        "will be made based on the evolution of inflows and the weather forecast for the "
        "next quarter."
    ),
    (
        "The new energy auction held this week traded approximately two thousand megawatts "
        "of capacity, distributed among wind, solar and small hydroelectric projects. The "
        "average contract price remained below the ceiling established by the regulatory "
        "agency, signaling that competition among generators remains intense. In the "
        "short-term market, the spot price of energy posted a sharp decline in recent days, "
        "following the improvement of hydrological conditions and the reduction of demand "
        "due to the extended holiday. Market analysts believe the movement may persist in "
        "the coming weeks if the rainfall forecasts materialize in the regions with the "
        "greatest generation weight. Distribution utilities, in turn, are monitoring the "
        "impact of price fluctuations on the final tariff paid by residential and commercial "
        "consumers, since part of the energy purchase costs is passed on automatically."
    ),
    (
        "Distributed generation of photovoltaic solar energy continues to grow at an "
        "accelerated pace, with more than three hundred thousand new systems connected to "
        "the grid in the last quarter, according to the survey released by the industry "
        "association. Growth was driven by falling prices of photovoltaic modules and rising "
        "conventional energy tariffs, which make the investment more attractive for homes "
        "and small businesses. The average payback period, which was about six years at the "
        "beginning of the decade, is now close to three and a half years in most regions. "
        "Experts warn, however, of the need to reinforce distribution networks in areas with "
        "the highest concentration of systems, avoiding overloads during high generation "
        "hours. Utilities have been modernizing their equipment and revising connection "
        "procedures to accommodate the new consumption and generation profile."
    ),
    (
        "The utility responsible for operating the transmission lines of the southern "
        "region announced that it will begin next Monday the scheduled maintenance of one "
        "of the main energy flow corridors, with an estimated duration of ten days. During "
        "this period, the transmission capacity of the segment will be reduced, and part of "
        "the energy will be redirected through alternative routes to guarantee supply to "
        "the cities served. The services include replacing insulators, thermographic "
        "inspection of the towers and clearing the easement strip, which presented "
        "vegetation growth after the rainy season. The company emphasizes that the works "
        "were planned to coincide with the period of lowest demand and that there is no "
        "forecast of supply interruption. Representatives of the regulatory agency will "
        "monitor the execution of the works and may apply sanctions in case of failure to "
        "meet the schedule."
    ),
    (
        "The arrival of a new cold front in the south-central region of the country should "
        "cause a sharp drop in temperatures in the coming days, with a direct impact on "
        "electricity consumption. According to the meteorological institute, the polar air "
        "mass advances from Thursday, bringing temperatures down in at least ten capitals "
        "and increasing the use of heaters in homes. The increase in demand during the "
        "night period, coinciding with peak hours, should raise the load of the power "
        "system by approximately four percent. The dispatch centers have already activated "
        "the most flexible thermal plants as a preventive measure and are monitoring load "
        "behavior in real time. Meteorologists say the episode should last about a week, "
        "followed by a gradual rise in temperatures. Utilities recommend attention to the "
        "simultaneous use of heating equipment during periods of higher demand."
    ),
]


def carregar_pares(
    arquivo_pt: Optional[str] = None,
    arquivo_en: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Retorna pares (nome, texto_pt, texto_en).

    Args:
        arquivo_pt/arquivo_en: caminhos opcionais com um documento por linha
            (se apenas um for dado, o outro usa o built-in correspondente).

    Returns:
        Lista de tuplas (nome_documento, texto_pt, texto_en).
    """
    if arquivo_pt or arquivo_en:
        pt_textos: List[str]
        en_textos: List[str]
        if arquivo_pt:
            pt_textos = [
                l.strip() for l in Path(arquivo_pt).read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
        else:
            pt_textos = list(AMOSTRAS_PT)
        if arquivo_en:
            en_textos = [
                l.strip() for l in Path(arquivo_en).read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
        else:
            en_textos = list(AMOSTRAS_EN)
        n = min(len(pt_textos), len(en_textos))
        return [(f"doc_{i+1}", pt_textos[i], en_textos[i]) for i in range(n)]

    return [
        (f"doc_{i+1}", pt, en)
        for i, (pt, en) in enumerate(zip(AMOSTRAS_PT, AMOSTRAS_EN))
    ]
