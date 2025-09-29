import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11

def load_final_training_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines[1:]:
        if line.strip() and '\t' in line:
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[0].isdigit():
                step = int(parts[0])
                loss = float(parts[1])
                data.append({'Step': step, 'Training_Loss': loss})
    
    return pd.DataFrame(data)

df_final = load_final_training_data('../data/finetuning-final.txt')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

ax1.plot(df_final['Step'], df_final['Training_Loss'], color='#2E86C1', alpha=0.8, linewidth=0.8)
ax1.set_xlabel('Passos de Treinamento')
ax1.set_ylabel('Perda de Treinamento')
ax1.set_title('Oscilações da Perda de Treinamento - Modelo Final (LR: 2e-4)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=df_final['Training_Loss'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Média: {df_final["Training_Loss"].mean():.3f}')
ax1.axhline(y=df_final['Training_Loss'].min(), color='green', linestyle='--', alpha=0.7, label=f'Mínimo: {df_final["Training_Loss"].min():.3f}')
ax1.legend()

window_size = 100
df_smooth = df_final['Training_Loss'].rolling(window=window_size).mean()

ax2.plot(df_final['Step'], df_smooth, color='#E74C3C', linewidth=2.5, label='Tendência Suavizada (Janela: 100)')
ax2.fill_between(df_final['Step'], df_smooth - df_final['Training_Loss'].rolling(window=window_size).std(), 
                 df_smooth + df_final['Training_Loss'].rolling(window=window_size).std(), 
                 alpha=0.2, color='#E74C3C', label='Desvio Padrão')

ax2.set_xlabel('Passos de Treinamento')
ax2.set_ylabel('Perda de Treinamento (Suavizada)')
ax2.set_title('Tendência de Convergência - Análise Estatística')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('../assets/training_loss_final_oscillations.png', dpi=300, bbox_inches='tight')
plt.show()

print("Análise das Oscilações do Training Loss:")
print(f"Perda Inicial: {df_final['Training_Loss'].iloc[0]:.4f}")
print(f"Perda Final: {df_final['Training_Loss'].iloc[-1]:.4f}")
print(f"Perda Mínima: {df_final['Training_Loss'].min():.4f}")
print(f"Perda Máxima: {df_final['Training_Loss'].max():.4f}")
print(f"Perda Média: {df_final['Training_Loss'].mean():.4f}")
print(f"Desvio Padrão: {df_final['Training_Loss'].std():.4f}")
print(f"Redução Total: {((df_final['Training_Loss'].iloc[0] - df_final['Training_Loss'].iloc[-1]) / df_final['Training_Loss'].iloc[0] * 100):.2f}%")
