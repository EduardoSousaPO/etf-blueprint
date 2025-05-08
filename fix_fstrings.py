#!/usr/bin/env python
"""
Este script corrige problemas de f-strings com caracteres de escape no arquivo resultados.py.
"""
import re
import os

def fix_fstrings(file_path):
    # Ler o arquivo
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Criar uma cópia de backup
    with open(file_path + '.bak', 'w', encoding='utf-8') as file:
        file.write(content)
    
    # Corrigir o CSS principal na função show()
    # Localizar blocos grandes de CSS com f-strings e substituí-los por strings normais
    pattern = r'(st\.markdown\(f"""\s*<style>.*?</style>\s*""",\s*unsafe_allow_html=True\))'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        # Substituir f""" por """ simples
        fixed = match.replace('st.markdown(f"""', 'st.markdown("""')
        content = content.replace(match, fixed)
    
    # Corrigir padrões típicos de strings com escape characters
    pattern = r'st\.markdown\(f""".*?\\.*?""",\s*unsafe_allow_html=True\)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        # Extrair o conteúdo da string
        content_pattern = r'st\.markdown\(f"""(.*?)""",\s*unsafe_allow_html=True\)'
        content_match = re.search(content_pattern, match, re.DOTALL)
        
        if content_match:
            string_content = content_match.group(1)
            
            # Verificar se há expressões de formatação
            format_vars = re.findall(r'\{([^{}]*)\}', string_content)
            
            if format_vars:
                # Criar uma versão com .format() normal
                for var in format_vars:
                    placeholder = '{' + var + '}'
                    string_content = string_content.replace(placeholder, '{{{}}}')
                
                # Substituir a string original
                new_string = f'st.markdown("""{string_content}""".format({", ".join(format_vars)}), unsafe_allow_html=True)'
                content = content.replace(match, new_string)
            else:
                # Se não há variáveis, apenas remover o 'f'
                new_string = match.replace('st.markdown(f"""', 'st.markdown("""')
                content = content.replace(match, new_string)
    
    # Salvar o arquivo corrigido
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Arquivo {file_path} corrigido e backup salvo como {file_path}.bak")

if __name__ == "__main__":
    file_path = "src/frontend/pages/resultados.py"
    if os.path.exists(file_path):
        fix_fstrings(file_path)
    else:
        print(f"Arquivo {file_path} não encontrado") 