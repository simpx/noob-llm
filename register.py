from transformers import AutoModelForCausalLM, AutoConfig
from model import Noob, NoobConfig

# 注册配置
AutoConfig.register("Noob", NoobConfig)
# 注册模型
AutoModelForCausalLM.register(NoobConfig, Noob)