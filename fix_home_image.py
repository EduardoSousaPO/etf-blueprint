#!/usr/bin/env python
"""
Este script corrige o problema com o parâmetro 'use_container_width' nas chamadas st.image e st.plotly_chart
da página home.py
"""
import os
import re

def fix_image_params():
    file_path = "src/frontend/pages/home.py"
    
    # Verificar se o arquivo existe
    if not os.path.exists(file_path):
        print(f"Arquivo {file_path} não encontrado!")
        return
    
    # Ler o conteúdo do arquivo
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Criar backup do arquivo original
    with open(f"{file_path}.bak", 'w', encoding='utf-8') as file:
        file.write(content)
        print(f"Backup criado como {file_path}.bak")
    
    # Substituir o parâmetro use_container_width para st.image
    updated_content = re.sub(
        r'st\.image\("assets/investment_chart\.png", use_container_width=True\)',
        r'st.image("assets/investment_chart.png")',
        content
    )
    
    # Substituir o parâmetro use_container_width para st.plotly_chart
    updated_content = re.sub(
        r'st\.plotly_chart\(fig, use_container_width=True\)',
        r'st.plotly_chart(fig)',
        updated_content
    )
    
    # Verificar se houve alteração
    if content == updated_content:
        print("Nenhuma alteração foi necessária, os padrões não foram encontrados.")
        
        # Procurar por padrões semelhantes, para diagnóstico
        all_images = re.findall(r'st\.image\(.*?\)', content)
        if all_images:
            print("Encontradas as seguintes chamadas de st.image:")
            for img in all_images:
                print(f" - {img}")
        
        all_plots = re.findall(r'st\.plotly_chart\(.*?\)', content)
        if all_plots:
            print("Encontradas as seguintes chamadas de st.plotly_chart:")
            for plot in all_plots:
                print(f" - {plot}")
        return
    
    # Salvar o arquivo corrigido
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)
        print(f"Arquivo {file_path} atualizado com sucesso!")
    
    # Procurar por outras chamadas com use_container_width
    # para ver se há mais casos a corrigir
    all_container_width = re.findall(r'use_container_width=True', updated_content)
    if all_container_width:
        print(f"ATENÇÃO: Ainda existem {len(all_container_width)} chamadas com use_container_width.")
        
        # Procurar especificamente em st.image
        all_images = re.findall(r'st\.image\(.*?use_container_width=True.*?\)', updated_content)
        if all_images:
            print("Chamadas st.image com use_container_width:")
            for img in all_images:
                print(f" - {img}")
        
        # Procurar especificamente em st.plotly_chart
        all_plots = re.findall(r'st\.plotly_chart\(.*?use_container_width=True.*?\)', updated_content)
        if all_plots:
            print("Chamadas st.plotly_chart com use_container_width:")
            for plot in all_plots:
                print(f" - {plot}")

if __name__ == "__main__":
    print("Iniciando correção da página home.py...")
    fix_image_params()
    print("Concluído!") 