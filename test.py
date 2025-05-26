import glob

#lora = sorted(glob.glob('./experiment/gpt*-lora_lr1e-3*.txt'))
#slora = sorted(glob.glob('./experiment/gpt*-slora_lr1e-3*.txt'))

#lora = sorted(glob.glob('./experiment/gptj-6b-lora-math-all-r32_*.txt'))
#slora = sorted(glob.glob('./experiment/gptj-6b-slora-math-all-r32_*.txt'))

lora = sorted(glob.glob('./experiment/gpt_math_10k_lora_r16_lr0.001_*.txt'))
slora = sorted(glob.glob('./experiment/gpt_math_10k_slora_r16_lr0.001_*.txt'))

assert len(lora) == len(slora)

print("LoRA vs. SLoRA")

lora_results = []
slora_results = []
for i in range(len(slora)):
    l = open(lora[i], 'r')
    sl = open(slora[i], 'r')
    
    print(lora[i].split('_')[-1].replace('.txt',''))

    lora_result = round(float(l.readlines()[-1].split(' ')[-1]),3)
    slora_result = round(float(sl.readlines()[-1].split(' ')[-1]),3)
    
    lora_results.append(lora_result)
    slora_results.append(slora_result)

    print(f'LoRA: {lora_result}')
    print(f'SLoRA: {slora_result}')

print(f'AVG LoRA: {sum(lora_results)/ len(lora_results)}')
print(f'AVG SLoRA: {sum(slora_results)/ len(slora_results)}')