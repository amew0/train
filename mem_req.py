from transformers import AutoModel; 
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; 

model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"
print(f"\nModel: {model_name_or_path}\n\n")
model = AutoModel.from_pretrained(model_name_or_path)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)




### Output
'''
Model: Qwen/Qwen2-VL-7B-Instruct

Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 7070M total params, 544M largest layer params.
  per CPU  |  per GPU |   Options
  177.80GB |   2.03GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
  177.80GB |   2.03GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
  158.04GB |  15.20GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
  158.04GB |  15.20GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
    3.05GB | 120.56GB | offload_param=none, offload_optimizer=none, zero_init=1
   39.51GB | 120.56GB | offload_param=none, offload_optimizer=none, zero_init=0



--------------------
Model: Qwen/Qwen2-VL-2B-Instruct

Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 1543M total params, 233M largest layer params.
  per CPU  |  per GPU |   Options
   38.82GB |   0.87GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
   38.82GB |   0.87GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
   34.50GB |   3.74GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
   34.50GB |   3.74GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
    1.30GB |  26.75GB | offload_param=none, offload_optimizer=none, zero_init=1
    8.63GB |  26.75GB | offload_param=none, offload_optimizer=none, zero_init=0

'''