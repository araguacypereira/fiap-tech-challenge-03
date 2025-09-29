import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

def load_finetuning_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines[2:]:
        if line.strip() and '\t' in line:
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[0].isdigit():
                step = int(parts[0])
                loss = float(parts[1])
                data.append({'Step': step, 'Training_Loss': loss})
    
    return pd.DataFrame(data)

df1 = load_finetuning_data('../data/finetuning-01.txt')
df2 = load_finetuning_data('../data/finetuning-02.txt')
df3 = load_finetuning_data('../data/finetuning-03.txt')

min_steps = min(len(df1), len(df2), len(df3))
window_size = 50

df1_smooth = df1[:min_steps]['Training_Loss'].rolling(window=window_size).mean()
df2_smooth = df2[:min_steps]['Training_Loss'].rolling(window=window_size).mean()
df3_smooth = df3[:min_steps]['Training_Loss'].rolling(window=window_size).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

ax1.plot(df1['Step'], df1['Training_Loss'], label='Taxa de Aprendizado: 2e-5', alpha=0.7, linewidth=1)
ax1.plot(df2['Step'], df2['Training_Loss'], label='Taxa de Aprendizado: 5e-5', alpha=0.7, linewidth=1)
ax1.plot(df3['Step'], df3['Training_Loss'], label='Taxa de Aprendizado: 1e-4', alpha=0.7, linewidth=1)

ax1.set_xlabel('Passos de Treinamento')
ax1.set_ylabel('Perda de Treinamento')
ax1.set_title('Comparação Fine-tuning: Dados Brutos')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(min_steps), df1_smooth, label='Taxa de Aprendizado: 2e-5 (suavizado)', linewidth=2.5)
ax2.plot(range(min_steps), df2_smooth, label='Taxa de Aprendizado: 5e-5 (suavizado)', linewidth=2.5)
ax2.plot(range(min_steps), df3_smooth, label='Taxa de Aprendizado: 1e-4 (suavizado)', linewidth=2.5)

ax2.set_xlabel('Passos de Treinamento')
ax2.set_ylabel('Perda de Treinamento (Suavizada)')
ax2.set_title('Comparação Fine-tuning: Tendência Suavizada (Média Móvel)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/finetuning_comparison_completo.png', dpi=300, bbox_inches='tight')
plt.show()

print("Resumo Estatístico:")
print(f"LR 2e-5: Perda Mínima = {df1['Training_Loss'].min():.4f}, Perda Final = {df1['Training_Loss'].iloc[-1]:.4f}")
print(f"LR 5e-5: Perda Mínima = {df2['Training_Loss'].min():.4f}, Perda Final = {df2['Training_Loss'].iloc[-1]:.4f}")
print(f"LR 1e-4: Perda Mínima = {df3['Training_Loss'].min():.4f}, Perda Final = {df3['Training_Loss'].iloc[-1]:.4f}")
