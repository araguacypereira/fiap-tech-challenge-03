import json

def converter_para_alpaca(input_file, output_file, num_registros=10000):
    """
    Converte os primeiros N registros de um arquivo JSON para o formato Alpaca.
    
    Args:
        input_file (str): Caminho para o arquivo JSON de entrada
        output_file (str): Caminho para o arquivo JSON de saída no formato Alpaca
        num_registros (int): Número de registros a serem convertidos (padrão: 10000)
    """
    
    dados_alpaca = []
    
    contador = 0
    
    print(f"Iniciando conversão dos primeiros {num_registros} registros...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for linha in f:
            if contador >= num_registros:
                break
                
            try:
                registro = json.loads(linha.strip())
                
                entrada_alpaca = {
                    "instruction": registro.get("title", ""),
                    "input": "",
                    "output": registro.get("content", "")
                }
                
                dados_alpaca.append(entrada_alpaca)
                contador += 1
                
                # Mostrar progresso a cada 1000 registros
                if contador % 1000 == 0:
                    print(f"Registros processados: {contador}")
                    
            except json.JSONDecodeError as e:
                print(f"Erro ao processar linha {contador + 1}: {e}")
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dados_alpaca, f, ensure_ascii=False, indent=2)
    
    print(f"Conversão concluída! {contador} registros foram salvos em {output_file}")

if __name__ == "__main__":
    arquivo_entrada = "amazon_data_single.json"
    arquivo_saida = "dataset_alpaca.json"
    
    converter_para_alpaca(arquivo_entrada, arquivo_saida, 1000)
