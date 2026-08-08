from .AutoTimes_Gpt2 import Model as AutoTimes_Gpt2
from .AutoTimes_Gpt2_SMETimes import Model as AutoTimes_Gpt2_SMETimes
from .AutoTimes_Llama import Model as AutoTimes_Llama
from .AutoTimes_Opt_1b import Model as AutoTimes_Opt_1b
from .AutoTimes_Qwen import Model as AutoTimes_Qwen
model_dict = {
    'AutoTimes_Gpt2': AutoTimes_Gpt2,
    'AutoTimes_Gpt2_SMETimes': AutoTimes_Gpt2_SMETimes,
    'AutoTimes_Llama': AutoTimes_Llama,
    'AutoTimes_Opt_1b': AutoTimes_Opt_1b,
    'AutoTimes_Qwen': AutoTimes_Qwen
}