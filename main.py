import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# 1. ORGANIZAÇÃO E DICIONÁRIO DE NOMES
# ==============================================================================
pasta_saida = "resultados_rmsd"
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# Este dicionário "traduz" o nome gigante do arquivo para o título bonito do gráfico
sistemas = {
    "folp1_AF_receptor__DDS_dapsona_complex": "FolP1 x WT",
    "folP1_Val14Gly_receptor__DDS_dapsona_complex": "FolP1 x Val14Gly",
    "folP1_Val14Ile_receptor__DDS_dapsona_complex": "FolP1 x Val14Ile",
    "folp2_AF_receptor__DDS_dapsona_complex": "FolP2 x WT",
    "folP2_Arg23Ser_receptor__DDS_dapsona_complex": "FolP2 x Arg23Ser",
    "folP2_Pro25Leu_receptor__DDS_dapsona_complex": "FolP2 x Pro25Leu"
}

replicas = [1, 2, 3]
cores = ['red', 'blue', 'green']

# Preparar a imagem "Mestre" com 6 blocos (3 linhas e 2 colunas)
fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, sharey=True)
axes = axes.flatten()  # Facilita o acesso a cada bloco do gráfico

# Criar e abrir o arquivo TXT
caminho_txt = os.path.join(pasta_saida, "estatisticas_rmsd.txt")

print("=== INICIANDO ANÁLISE EM LOTE ===")

# ==============================================================================
# 2. PROCESSAMENTO E GERAÇÃO DOS DADOS
# ==============================================================================
with open(caminho_txt, "w", encoding="utf-8") as arquivo_txt:
    arquivo_txt.write("=== RELATÓRIO ESTATÍSTICO DE RMSD (Em Angstroms) ===\n\n")

    # Faz o loop por cada sistema do dicionário
    for idx, (sistema_longo, nome_curto) in enumerate(sistemas.items()):
        print(f"➤ Processando: {nome_curto}...")
        arquivo_txt.write(f"SISTEMA: {nome_curto}\n")
        arquivo_txt.write("-" * 40 + "\n")

        ax = axes[idx]  # Seleciona o bloco correto do painel para desenhar
        topologia = f"dados/{sistema_longo}.prmtop"

        # Loop pelas réplicas de cada sistema
        for i, rep in enumerate(replicas):
            trajetoria = f"dados/{sistema_longo}_rep{rep}.dcd"

            try:
                # Carregar e calcular
                u = mda.Universe(topologia, trajetoria)
                rmsd_analise = rms.RMSD(u, u, select='backbone')
                rmsd_analise.run()

                # Extrair dados e converter para ns
                frames = rmsd_analise.results.rmsd.T[1]
                valores_rmsd = rmsd_analise.results.rmsd.T[2]
                tempo_ns = frames * 0.001

                # Plotar a linha dentro do bloco específico
                ax.plot(tempo_ns, valores_rmsd, color=cores[i], linewidth=1.0, label=f'Réplica {rep}', alpha=0.8)

                # Estatísticas
                rmsd_medio = np.mean(valores_rmsd)
                rmsd_max = np.max(valores_rmsd)
                rmsd_dp = np.std(valores_rmsd)

                # Escrever no TXT
                arquivo_txt.write(f"  Réplica {rep}:\n")
                arquivo_txt.write(f"    - RMSD Médio:  {rmsd_medio:.2f} ± {rmsd_dp:.2f} Å\n")
                arquivo_txt.write(f"    - RMSD Máximo: {rmsd_max:.2f} Å\n")

            except Exception as e:
                print(f"  [ERRO] Falha na {nome_curto} Rep {rep}. Verifique o arquivo.")
                arquivo_txt.write(f"  Réplica {rep}: ERRO AO PROCESSAR\n")

        # ==============================================================================
        # 3. FORMATAÇÃO DE CADA BLOCO DO PAINEL
        # ==============================================================================
        ax.set_title(nome_curto, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="lower right", fontsize=8)

        # Colocar etiquetas (X e Y) apenas nas bordas do painel para ficar limpo
        if idx >= 4:  # Última linha de gráficos
            ax.set_xlabel("Tempo (ns)", fontsize=11)
        if idx % 2 == 0:  # Primeira coluna de gráficos
            ax.set_ylabel("RMSD (Å)", fontsize=11)

        arquivo_txt.write("\n")

# ==============================================================================
# 4. SALVAR IMAGEM E FINALIZAR
# ==============================================================================
# Ajustar os espaços para os títulos não se sobreporem
plt.tight_layout()

# Salvar o painel único com alta resolução
caminho_imagem = os.path.join(pasta_saida, "painel_rmsd_completo.png")
plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')

print(f"\n=== SUCESSO! ===")
print(f"1. Painel gerado: {caminho_imagem}")
print(f"2. Dados estatísticos salvos em: {caminho_txt}")
plt.show()