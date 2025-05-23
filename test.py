import glob

#lora = sorted(glob.glob('./experiment/gpt*-lora_lr1e-3*.txt'))
#slora = sorted(glob.glob('./experiment/gpt*-slora_lr1e-3*.txt'))

#lora = sorted(glob.glob('./experiment/gptj-6b-lora-math-all-r32_*.txt'))
#slora = sorted(glob.glob('./experiment/gptj-6b-slora-math-all-r32_*.txt'))

lora = sorted(glob.glob('./experiment/gpt_math_50k_lora_r32_lr0.0003_*.txt'))
slora = sorted(glob.glob('./experiment/gpt_math_50k_slora_r32_lr0.0003_*.txt'))

assert len(lora) == len(slora)

print("LoRA vs. SLoRA")

for i in range(len(lora)):
    l = open(lora[i], 'r')
    sl = open(slora[i], 'r')
    
    print(lora[i].split('_')[-1].replace('.txt',''))
    print(f'LoRA: {l.readlines()[-1]}')
    print(f'SLoRA: {sl.readlines()[-1]}')