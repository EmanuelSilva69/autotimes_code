"""Memoria episodica do Self-Refine: JSON tolerante, gravacao atomica, FIFO.

Estado FORA do Git (`.runtime/autotimes_memory.json`, gitignorado) porque e
reprodutivel: e historico de falhas/correcoes, nao resultado de experimento.

Contratos (spec self-refine-textual-generation):
    - ausente / vazio / corrompido -> memoria vazia + AVISO, sem excecao;
    - escrita ATOMICA (tmp + os.replace): uma interrupcao nao corrompe o
      arquivo que ja estava la;
    - FIFO com teto (`max_entradas`): nao cresce sem limite.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

TETO_ENTRADAS: int = 200
PREFIXO_AVISO: str = "[self-refine][memoria] AVISO:"


@dataclass
class Entrada:
    """Uma licao: o que falhou, como corrigir e em que contexto."""

    quando: str
    codigo: str
    motivo: str
    regra: str
    contexto: Dict[str, Any] = field(default_factory=dict)
    registro: str = ""  # prompt de refinamento renderizado (registro, D9)

    def como_feedback(self) -> Dict[str, Any]:
        return {"codigo": self.codigo, "motivo": self.motivo,
                "regra": self.regra}


def _aviso(msg: str) -> None:
    print(f"{PREFIXO_AVISO} {msg}")


class Memoria:
    """Historico de falhas/correcoes persistido em JSON."""

    def __init__(self, caminho: Path, max_entradas: int = TETO_ENTRADAS):
        self.caminho = Path(caminho)
        self.max_entradas = int(max_entradas)
        self.entradas: List[Entrada] = []
        self._carregado = False

    # ---- leitura tolerante ----------------------------------------- #
    def carregar(self) -> List[Entrada]:
        """Le o arquivo; qualquer problema -> [] + aviso (nunca excecao)."""
        self._carregado = True
        p = self.caminho
        if not p.exists():
            self.entradas = []
            return self.entradas
        try:
            texto = p.read_text(encoding="utf-8")
        except OSError as e:
            _aviso(f"nao consegui ler {p} ({e}); comecando com memoria vazia")
            self.entradas = []
            return self.entradas
        if not texto.strip():
            _aviso(f"{p} vazio; comecando com memoria vazia")
            self.entradas = []
            return self.entradas
        try:
            dados = json.loads(texto)
        except json.JSONDecodeError as e:
            _aviso(f"{p} corrompido ({e}); comecando com memoria vazia")
            self.entradas = []
            return self.entradas
        brutas = dados.get("entradas", []) if isinstance(dados, dict) else []
        entradas = []
        for item in brutas:
            if not isinstance(item, dict):
                continue
            try:
                entradas.append(Entrada(
                    quando=str(item.get("quando", "")),
                    codigo=str(item.get("codigo", "")),
                    motivo=str(item.get("motivo", "")),
                    regra=str(item.get("regra", "")),
                    contexto=dict(item.get("contexto", {}) or {}),
                    registro=str(item.get("registro", ""))[:2000],
                ))
            except (TypeError, ValueError):
                continue  # entrada isolada ruim nao derruba o arquivo inteiro
        self.entradas = entradas[-self.max_entradas:]
        return self.entradas

    def _garantir(self) -> List[Entrada]:
        if not self._carregado:
            return self.carregar()
        return self.entradas

    # ---- consulta ---------------------------------------------------- #
    def consultar(self, contexto: Optional[Dict[str, Any]] = None,
                  limite: int = 5) -> List[Entrada]:
        """Licoes mais relevantes (as ultimas, ou as que batem o contexto)."""
        entradas = self._garantir()
        if contexto:
            alvo = {k: v for k, v in contexto.items() if v is not None}
            if alvo:
                iguais = [e for e in entradas
                          if any(e.contexto.get(k) == v
                                 for k, v in alvo.items())]
                if iguais:
                    entradas = iguais
        return entradas[-limite:]

    def resumo(self, contexto: Optional[Dict[str, Any]] = None,
               limite: int = 3) -> str:
        """Licoes em uma linha cada (entra no <context_analysis> do prompt)."""
        licoes = self.consultar(contexto, limite=limite)
        if not licoes:
            return "nenhuma lição registrada ainda"
        return "\n".join(f"- {e.codigo}: {e.motivo} -> {e.regra}"
                         for e in licoes)

    # ---- escrita atomica -------------------------------------------- #
    def registrar(self, codigo: str, motivo: str, regra: str,
                  contexto: Optional[Dict[str, Any]] = None,
                  registro: str = "") -> Entrada:
        """Acrescenta uma licao e persiste de forma atomica (caminho de falha)."""
        entradas = self._garantir()
        entrada = Entrada(
            quando=datetime.now().isoformat(timespec="seconds"),
            codigo=str(codigo), motivo=str(motivo), regra=str(regra),
            contexto=dict(contexto or {}),
            registro=str(registro)[:2000],
        )
        entradas.append(entrada)
        self.entradas = entradas[-self.max_entradas:]
        self.salvar()
        return entrada

    def salvar(self) -> Path:
        """Escrita atomica: .tmp + os.replace (nunca deixa JSON pela metade)."""
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "versao": 1,
            "teto": self.max_entradas,
            "entradas": [asdict(e) for e in self.entradas],
        }
        tmp = self.caminho.with_suffix(self.caminho.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.caminho)  # atomico no mesmo filesystem
        return self.caminho
