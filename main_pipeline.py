from classification import classificar_e_avaliar
from clustering import clusterizar_celulares
from summarization import resumir_clusters

if __name__ == "__main__":
    # Caminho para a base original
    input_file = "base_roubo.csv"

    # Etapa 1: Classificação e Avaliação com SVM
    classificar_e_avaliar(input_file)

    # Etapa 2: Clusterização de relatos de celular
    cluster_csv = clusterizar_celulares(input_file)

    # Etapa 3: Sumarização com LangChain
    resumo = resumir_clusters(cluster_csv)
    print("\n\n📌 RESUMO FINAL GERADO PELO MODELO:\n")
    print(resumo)