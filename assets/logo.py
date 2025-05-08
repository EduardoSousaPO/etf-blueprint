"""
Script para gerar logo do ETF Blueprint usando Matplotlib
Para gerar a logo, execute python assets/logo.py
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as PathEffects
import os

# Configurar o fundo
plt.figure(figsize=(8, 4))
ax = plt.axes()
ax.set_facecolor('#f0f2f6')

# Gerar pontos para a fronteira eficiente
x = np.linspace(5, 25, 100)
y = 2 + 0.4 * x + 0.4 * np.sin(x/5)

# Desenhar a fronteira eficiente
plt.plot(x, y, 'b-', linewidth=4, alpha=0.7)

# Adicionar pontos
plt.scatter([8, 15, 20], [5, 9, 12], 
            c=['blue', 'green', 'red'], 
            s=[150, 200, 150], 
            zorder=5)

# Adicionar título
txt = plt.text(15, 15, 'ETF Blueprint', 
              fontsize=40, 
              fontweight='bold', 
              ha='center',
              color='#1e3a8a')

# Adicionar sombra ao texto
txt.set_path_effects([
    PathEffects.withStroke(linewidth=3, foreground='white')
])

# Adicionar subtítulo
plt.text(15, 13, 'Carteiras Otimizadas de ETFs', 
        fontsize=14, 
        ha='center',
        color='#333333')

# Remover eixos
plt.xticks([])
plt.yticks([])
plt.axis('off')

# Criar diretório de assets se não existir
os.makedirs('assets', exist_ok=True)

# Salvar imagem
plt.savefig('assets/logo.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.savefig('assets/logo_small.png', dpi=150, bbox_inches='tight', pad_inches=0.1)

# Gerar imagem para a home
plt.figure(figsize=(10, 6))
ax = plt.axes()
ax.set_facecolor('#f0f2f6')

# Dados para o gráfico de pizza
labels = ['Ações EUA', 'Ações Internacionais', 'Mercados Emergentes', 
          'Renda Fixa', 'Imobiliário', 'Commodities']
sizes = [40, 20, 10, 20, 5, 5]
colors = plt.cm.Paired(np.arange(len(labels))/len(labels))

# Gráfico de pizza
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
plt.axis('equal')

# Título
plt.title('Exemplo de Alocação Diversificada', fontsize=20, pad=20, color='#1e3a8a')

# Salvar imagem para a home
plt.savefig('assets/investment_chart.png', dpi=150, bbox_inches='tight')

print("Imagens geradas com sucesso!") 