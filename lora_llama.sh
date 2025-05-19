CUDA_VISIBLE_DEVICES=$1 python finetune.py --base_model 'yahma/llama-7b-hf' --data_path './ft-training_set/math_10k.json' --output_dir './trained_models/llama-7b-lora-math_1e-3/'   --batch_size 16  --micro_batch_size 4   --num_epochs 3   --learning_rate 1e-3   --cutoff_len 256   --val_set_size 0 --eval_step 80 --save_step 80  --adapter_name lora --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' --lora_r 32 --lora_alpha 64

CUDA_VISIBLE_DEVICES=$1 python evaluate.py --model LLaMA-7B --adapter LoRA --dataset SVAMP --base_model 'yahma/llama-7b-hf' --lora_weights './trained_models/llama-7b-lora-math_1e-3'
CUDA_VISIBLE_DEVICES=$1 python evaluate.py --model LLaMA-7B --adapter LoRA --dataset AQuA --base_model 'yahma/llama-7b-hf' --lora_weights './trained_models/llama-7b-lora-math_1e-3'
CUDA_VISIBLE_DEVICES=$1 python evaluate.py --model LLaMA-7B --adapter LoRA --dataset AddSub --base_model 'yahma/llama-7b-hf' --lora_weights './trained_models/llama-7b-lora-math_1e-3'
CUDA_VISIBLE_DEVICES=$1 python evaluate.py --model LLaMA-7B --adapter LoRA --dataset gsm8k --base_model 'yahma/llama-7b-hf' --lora_weights './trained_models/llama-7b-lora-math_1e-3'
CUDA_VISIBLE_DEVICES=$1 python evaluate.py --model LLaMA-7B --adapter LoRA --dataset MultiArith --base_model 'yahma/llama-7b-hf' --lora_weights './trained_models/llama-7b-lora-math_1e-3'
CUDA_VISIBLE_DEVICES=$1 python evaluate.py --model LLaMA-7B --adapter LoRA --dataset SingleEq --base_model 'yahma/llama-7b-hf' --lora_weights './trained_models/llama-7b-lora-math_1e-3'
