import MDAnalysis as mda
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# 1. ORGANIZAÇÃO E VARIÁVEIS DO SISTEMA
# ==============================================================================
pasta_saida = "resultados_distancia"
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

nome_ligante = "DDS"

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

caminho_txt = os.path.join(pasta_saida, "estatisticas_distancia.txt")

print(f"=== INICIANDO ANÁLISE DE DISTÂNCIA MÍNIMA (Proteína vs {nome_ligante}) ===")

# ==============================================================================
# 2. PROCESSAMENTO E GERAÇÃO DOS DADOS
# ==============================================================================
with open(caminho_txt, "w", encoding="utf-8") as arquivo_txt:
    arquivo_txt.write("=== RELATÓRIO ESTATÍSTICO DE DISTÂNCIA MÍNIMA (Em Angstroms) ===\n")
    arquivo_txt.write(f"Mede o contacto mais próximo entre a Proteína e a Dapsona.\n\n")

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
                proteina = u.select_atoms('protein')
                dapsona = u.select_atoms(f'resname {nome_ligante}')

                distancias_minimas = []

                # Avança frame a frame calculando a distância matemática exata
                for ts in u.trajectory:
                    # Cria uma matriz com a distância de todos os átomos da Dapsona para a Proteína
                    matriz_dist = distances.distance_array(dapsona.positions, proteina.positions)
                    # Guarda apenas a distância mais curta (o contacto direto)
                    distancias_minimas.append(np.min(matriz_dist))

                distancias_minimas = np.array(distancias_minimas)
                tempo_ns = np.linspace(0, 100, len(distancias_minimas))

                ax.plot(tempo_ns, distancias_minimas, color=cores[i], linewidth=1.0, label=f'Réplica {rep}', alpha=0.8)

                dist_media = np.mean(distancias_minimas)

                arquivo_txt.write(f"  Réplica {rep}:\n")
                arquivo_txt.write(f"    - Distância Média de Contacto: {dist_media:.2f} Å\n")

                print(f"  [OK] Réplica {rep} | Distância Média: {dist_media:.2f} Å")

            except Exception as e:
                print(f"  [ERRO] Falha na {nome_curto} Rep {rep}. Motivo: {e}")
                arquivo_txt.write(f"  Réplica {rep}: ERRO AO PROCESSAR\n")

        # ==============================================================================
        # 3. FORMATAÇÃO DO BLOCO DO PAINEL
        # ==============================================================================
        ax.set_title(nome_curto, fontsize=13, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper left", fontsize=8)

        if idx >= 4:
            ax.set_xlabel("Tempo (ns)", fontsize=11)
        if idx % 2 == 0:
            ax.set_ylabel("Distância (Å)", fontsize=11)

        arquivo_txt.write("\n")

# ==============================================================================
# 4. SALVAR E FINALIZAR
# ==============================================================================
plt.tight_layout(h_pad=2.0, w_pad=2.0)

caminho_imagem = os.path.join(pasta_saida, "painel_distancia_completo.png")
plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')

print(f"\n=== SUCESSO! ===")
print(f"1. Painel gerado: {caminho_imagem}")
print(f"2. Dados estatísticos salvos em: {caminho_txt}")
plt.show()