import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# 1. ORGANIZAÇÃO E DICIONÁRIO DE NOMES
# ==============================================================================
pasta_saida = "resultados_rmsf"
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

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

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

caminho_txt = os.path.join(pasta_saida, "estatisticas_rmsf.txt")

print("=== INICIANDO ANÁLISE DE RMSF (FLUTUAÇÃO POR RESÍDUO) EM LOTE ===")

# ==============================================================================
# 2. PROCESSAMENTO E GERAÇÃO DOS DADOS
# ==============================================================================
with open(caminho_txt, "w", encoding="utf-8") as arquivo_txt:
    arquivo_txt.write("=== RELATÓRIO ESTATÍSTICO DE RMSF (Em Angstroms) ===\n\n")

    for idx, (sistema_longo, nome_curto) in enumerate(sistemas.items()):
        print(f"\n➤ Processando: {nome_curto}...")
        arquivo_txt.write(f"SISTEMA: {nome_curto}\n")
        arquivo_txt.write("-" * 40 + "\n")

        ax = axes[idx]
        topologia = f"dados/{sistema_longo}.prmtop"

        for i, rep in enumerate(replicas):
            trajetoria = f"dados/{sistema_longo}_rep{rep}.dcd"

            try:
                # 2.1 Carregar ficheiros
                u = mda.Universe(topologia, trajetoria)

                # 2.2 O RMSF é calculado usando apenas os Carbonos Alfa (CA)
                # de cada aminoácido para medir o esqueleto da proteína
                calphas = u.select_atoms('protein and name CA')

                # 2.3 Calcular RMSF
                rmsf_analise = rms.RMSF(calphas)
                rmsf_analise.run()

                valores_rmsf = rmsf_analise.results.rmsf
                residuos = calphas.resnums  # Obtém o número exato do aminoácido

                # 2.4 Plotar a linha no bloco específico
                ax.plot(residuos, valores_rmsf, color=cores[i], linewidth=1.2, label=f'Réplica {rep}', alpha=0.8)

                # 2.5 Estatísticas enriquecidas (descobre qual é o resíduo mais flexível)
                rmsf_medio = np.mean(valores_rmsf)
                rmsf_max = np.max(valores_rmsf)
                idx_max = np.argmax(valores_rmsf)
                residuo_max = residuos[idx_max]  # Identifica qual o resíduo que atingiu o pico

                # Escrever no TXT
                arquivo_txt.write(f"  Réplica {rep}:\n")
                arquivo_txt.write(f"    - RMSF Médio global: {rmsf_medio:.2f} Å\n")
                arquivo_txt.write(f"    - Pico Máximo:       {rmsf_max:.2f} Å (Ocorrido no Resíduo {residuo_max})\n")

                print(f"  [OK] Réplica {rep} | RMSF Médio: {rmsf_medio:.2f} Å | Pico no Resíduo {residuo_max}")

            except Exception as e:
                print(f"  [ERRO] Falha na {nome_curto} Rep {rep}. Motivo: {e}")
                arquivo_txt.write(f"  Réplica {rep}: ERRO AO PROCESSAR\n")

        # ==============================================================================
        # 3. FORMATAÇÃO DO BLOCO DO PAINEL
        # ==============================================================================
        ax.set_title(nome_curto, fontsize=13, fontweight='bold')

        # Limita o eixo X do menor para o maior resíduo
        ax.set_xlim(min(residuos), max(residuos))

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

        # Etiquetas dos eixos (X agora é o Número do Resíduo)
        if idx >= 4:
            ax.set_xlabel("Número do Resíduo", fontsize=11)
        if idx % 2 == 0:
            ax.set_ylabel("RMSF (Å)", fontsize=11)

        arquivo_txt.write("\n")

# ==============================================================================
# 4. SALVAR E FINALIZAR
# ==============================================================================
# Ajusta o espaçamento para ficar limpo
plt.tight_layout(h_pad=2.0, w_pad=2.0)

# Salvar a imagem final
caminho_imagem = os.path.join(pasta_saida, "painel_rmsf_completo.png")
plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')

print(f"\n=== SUCESSO! ===")
print(f"1. Painel gerado: {caminho_imagem}")
print(f"2. Dados estatísticos salvos em: {caminho_txt}")
plt.show()