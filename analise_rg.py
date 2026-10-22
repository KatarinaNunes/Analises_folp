import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# 1. ORGANIZAÇÃO E DICIONÁRIO DE NOMES
# ==============================================================================
pasta_saida = "resultados_rg"
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

caminho_txt = os.path.join(pasta_saida, "estatisticas_rg.txt")

print("=== INICIANDO ANÁLISE DE RAIO DE GIRO (Rg) EM LOTE ===")

# ==============================================================================
# 2. PROCESSAMENTO E GERAÇÃO DOS DADOS
# ==============================================================================
with open(caminho_txt, "w", encoding="utf-8") as arquivo_txt:
    arquivo_txt.write("=== RELATÓRIO ESTATÍSTICO DE RAIO DE GIRO (Em Angstroms) ===\n\n")

    for idx, (sistema_longo, nome_curto) in enumerate(sistemas.items()):
        print(f"\n➤ Processando: {nome_curto}...")
        arquivo_txt.write(f"SISTEMA: {nome_curto}\n")
        arquivo_txt.write("-" * 40 + "\n")

        ax = axes[idx]
        topologia = f"dados/{sistema_longo}.prmtop"

        for i, rep in enumerate(replicas):
            trajetoria = f"dados/{sistema_longo}_rep{rep}.dcd"

            try:
                u = mda.Universe(topologia, trajetoria)
                proteina = u.select_atoms('backbone')

                valores_rg = []

                # Conta os frames
                for ts in u.trajectory:
                    valores_rg.append(proteina.radius_of_gyration())

                valores_rg = np.array(valores_rg)

                # CORREÇÃO: Distribui os valores recolhidos ao longo de exatos 100 ns
                tempo_ns = np.linspace(0, 100, len(valores_rg))

                ax.plot(tempo_ns, valores_rg, color=cores[i], linewidth=1.0, label=f'Réplica {rep}', alpha=0.8)

                rg_medio = np.mean(valores_rg)
                rg_max = np.max(valores_rg)
                rg_dp = np.std(valores_rg)

                arquivo_txt.write(f"  Réplica {rep}:\n")
                arquivo_txt.write(f"    - Rg Médio:  {rg_medio:.2f} ± {rg_dp:.2f} Å\n")
                arquivo_txt.write(f"    - Rg Máximo: {rg_max:.2f} Å\n")

                print(f"  [OK] Réplica {rep} | Rg Médio: {rg_medio:.2f} Å")

            except Exception as e:
                print(f"  [ERRO] Falha na {nome_curto} Rep {rep}. Motivo: {e}")
                arquivo_txt.write(f"  Réplica {rep}: ERRO AO PROCESSAR\n")

        # ==============================================================================
        # 3. FORMATAÇÃO DO BLOCO DO PAINEL
        # ==============================================================================
        ax.set_title(nome_curto, fontsize=13, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="best", fontsize=8)

        if idx >= 4:
            ax.set_xlabel("Tempo (ns)", fontsize=11)
        if idx % 2 == 0:
            ax.set_ylabel("Raio de Giro (Å)", fontsize=11)

        arquivo_txt.write("\n")

# ==============================================================================
# 4. SALVAR E FINALIZAR
# ==============================================================================
# Ajusta o espaçamento entre os gráficos (h_pad) para os títulos não colarem!
plt.tight_layout(h_pad=2.0, w_pad=2.0)

caminho_imagem = os.path.join(pasta_saida, "painel_rg_completo.png")
plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')

print(f"\n=== SUCESSO! ===")
print(f"1. Painel gerado: {caminho_imagem}")
print(f"2. Dados estatísticos salvos em: {caminho_txt}")
plt.show()