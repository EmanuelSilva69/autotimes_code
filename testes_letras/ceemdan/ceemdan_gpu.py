"""ceemdan_gpu - Backend HIP (AMD/ROCm) para CEEMDAN via DLL + ctypes.

Usa ``CEEMDAN-GPU/ceemdan_gpu.dll`` (port HIP do EEGLab-Pannon/CEEMDAN-GPU,
compilado para gfx1200 / RX 9060 XT) - ver ``valida_gpu.py`` para validacao.

Convencoes (iguais ao PyEMD usado pelo backend CPU):
  - ``ceemdan_gpu(x, max_imf=k)`` retorna ``cIMFs`` com ``k + 1`` linhas:
    linha 0 = IMF de maior frequencia, linhas 1..k-1 = oscilacoes,
    ultima linha = resíduo final (tendencia);
  - mesma escala do sinal de entrada (a saida crua da DLL e dividida por
    std(x, ddof=1) dentro da implementacao, pois o algoritmo roda sobre x/std);
  - float64 na saida.

Diferencas vs backend CPU (aceitas):
  - curand != numpy: a mesma seed NAO reproduz o mesmo ruído do PyEMD;
  - criterio de parada do sifting e a spline diferem do PyEMD;
  - semente e aceita como int (passada como unsigned long long).

Uso tipico::

    from testes_letras.ceemdan.ceemdan_gpu import ceemdan_gpu
    cIMFs = ceemdan_gpu(x, trials=100, epsilon=0.005, max_imf=8, seed=42)
"""

import ctypes
import os
from typing import Optional

import numpy as np

_DLL: Optional[ctypes.CDLL] = None
_DLL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "CEEMDAN-GPU", "ceemdan_gpu.dll"))
_ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.1\bin"
_DLL_DIR_HANDLE = None  # mantem vivo o add_dll_directory (py3.8+)

# amostras suficientes para EMD (espelha o guard do backend CPU)
_MIN_L = 10

# max_iter do sifting (valor usado pelos samples do repo)
_MAX_ITER = 100


def _carregar_dll() -> ctypes.CDLL:
    """Carrega a DLL uma unica vez (cache a nivel de modulo)."""
    global _DLL
    if _DLL is None:
        if not os.path.isfile(_DLL_PATH):
            raise FileNotFoundError(
                f"DLL HIP nao encontrada: {_DLL_PATH} (compile com hipcc)")
        global _DLL_DIR_HANDLE
        if os.path.isdir(_ROCM_BIN):
            if hasattr(os, "add_dll_directory"):
                _DLL_DIR_HANDLE = os.add_dll_directory(_ROCM_BIN)
            os.environ["PATH"] = _ROCM_BIN + os.pathsep + os.environ.get("PATH", "")
        dll = ctypes.CDLL(_DLL_PATH)
        dll.run_ceemdan_gpu.argtypes = [
            ctypes.c_int,              # numNoise (trials)
            ctypes.c_int,              # signalLength
            ctypes.c_int,              # numIMFs (max_imf + 1, com resíduo)
            ctypes.c_int,              # max_iter
            ctypes.POINTER(ctypes.c_float),  # input  (host)
            ctypes.POINTER(ctypes.c_float),  # output (host) numIMFs*L
            ctypes.c_float,            # noiseStrength (epsilon)
            ctypes.c_ulonglong,        # seed (curand)
        ]
        dll.run_ceemdan_gpu.restype = ctypes.c_double  # segundos no GPU
        _DLL = dll
    return _DLL


def ceemdan_gpu(x: np.ndarray, trials: int = 100, epsilon: float = 0.005,
                max_imf: int = 5, seed: int = 42) -> np.ndarray:
    """CEEMDAN na GPU (HIP) -> cIMFs (max_imf + 1, L) float64.

    Args:
        x: janela (L,) - convertida para float32 contiguo.
        trials: nº de realizacoes de ruido por IMF (numNoise).
        epsilon: amplitude do ruido adaptativo (noiseStrength).
        max_imf: nº maximo de IMFs; retorna max_imf+1 linhas (com resíduo).
        seed: semente curand.

    Returns:
        cIMFs (max_imf+1, L) float64 na escala de x.
        Linha 0 = IMF1 (alta freq), ultima linha = resíduo/tendencia.
    """
    x64 = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    L = x64.size

    if L < _MIN_L:
        cimfs = np.zeros((max_imf + 1, L), dtype=np.float64)
        cimfs[0] = x64
        return cimfs

    std = float(np.std(x64, ddof=1))
    if not np.isfinite(std) or std == 0.0:
        cimfs = np.zeros((max_imf + 1, L), dtype=np.float64)
        cimfs[0] = x64
        return cimfs

    dll = _carregar_dll()
    x32 = np.ascontiguousarray(x64, dtype=np.float32)
    out = np.empty((max_imf + 1, L), dtype=np.float32)

    elapsed = dll.run_ceemdan_gpu(
        ctypes.c_int(trials),
        ctypes.c_int(L),
        ctypes.c_int(max_imf + 1),
        ctypes.c_int(_MAX_ITER),
        x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(float(epsilon)),
        ctypes.c_ulonglong(int(seed)),
    )
    if elapsed < 0.0:
        raise RuntimeError(f"run_ceemdan_gpu falhou (allocacao GPU, t={elapsed})")

    # a DLL opera sobre x/std(x, ddof=1): devolve para a escala original
    imfs = out.astype(np.float64) * std
    if not np.all(np.isfinite(imfs)):
        # sinal tao curto/degenerado que o sifting nao converge (ex.: L=12):
        # fallback igual ao backend CPU (L<10): residuo = proprio sinal
        cimfs = np.zeros((max_imf + 1, L), dtype=np.float64)
        cimfs[0] = x64
        return cimfs
    return imfs


def decompor_ceemdan_gpu(
    x_window: np.ndarray,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
):
    """CEEMDAN GPU puro -> (tendencia, sazonalidade, residuos).

    Espelho de ``decomposicao.decompor_ceemdan`` (mesma convencao de linhas:
    cIMFs[0] = alta frequencia -> residuo; cIMFs[1:-1] -> sazonalidade;
    cIMFs[-1] -> tendencia).
    """
    x = np.ascontiguousarray(np.asarray(x_window, dtype=np.float64).ravel())
    L = x.size

    if L < _MIN_L:
        return x.copy(), np.zeros_like(x), np.zeros_like(x)

    cIMFs = ceemdan_gpu(x, trials=trials, epsilon=epsilon,
                        max_imf=max_imf, seed=seed)

    n_imfs = cIMFs.shape[0]
    if n_imfs <= 1 or not np.all(np.isfinite(cIMFs)):
        return x.copy(), np.zeros_like(x), np.zeros_like(x)

    residuo = cIMFs[0]
    if n_imfs > 2:
        sazonalidade = np.sum(cIMFs[1:-1], axis=0)
    else:
        sazonalidade = np.zeros_like(x)
    tendencia = cIMFs[-1]

    return tendencia, sazonalidade, residuo
