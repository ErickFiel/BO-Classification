import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Reutiliza a mesma função de rotulagem usada em classificacao.py

def preprocess(text):
    if pd.isnull(text):
        return ''
    text = str(text)
    text = re.sub(r'\W', ' ', text.lower())
    return ' '.join(text.split())

def search_and_label(input_file, column_name):
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None

    if column_name not in df.columns:
        print(f"A coluna '{column_name}' não foi encontrada no arquivo.")
        return None

    normalized_column = (
        df[column_name].astype(str)
        .str.strip()
        .str.lower()
        .fillna("")
        .apply(lambda x: re.sub(r'[^\w\s/]', '', x))
    )

    def classify_label(value):
        if "celular" in value:
            return 1
        elif value == "transeunte":
            return 0
        elif value == "moto":
            return 2
        elif value == "em coletivo":
            return 3
        else:
            return 4

    df['label'] = normalized_column.apply(classify_label)
    return df

def clusterizar_celulares(input_file: str) -> str:
    column_name = "especificacao_crime"
    result_df = search_and_label(input_file, column_name)
    if result_df is None:
        return None

    celular_df = result_df[result_df['label'] == 1].copy()
    print(f"Total de registros com label 'celular': {len(celular_df)}")

    celular_df['relato_limpo'] = celular_df['relato'].fillna('').apply(preprocess)

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(celular_df['relato_limpo'])

    # Cotovelo
    print("\n🔍 Aplicando método do cotovelo...")
    inertias = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, marker='o')
    plt.title('Método do Cotovelo - Inércia por número de clusters')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Inércia')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    knee = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
    ideal_k = knee.knee or 3
    print(f"\n✅ Número ideal de clusters detectado: {ideal_k}")

    kmeans_final = KMeans(n_clusters=ideal_k, random_state=42, n_init='auto')
    celular_df['cluster'] = kmeans_final.fit_predict(X)

    print("\n📊 Distribuição dos clusters:")
    print(celular_df['cluster'].value_counts())

    for cluster_id in sorted(celular_df['cluster'].unique()):
        print(f"\n📂 Cluster {cluster_id} - Top 3 relatos:")
        print(celular_df[celular_df['cluster'] == cluster_id]['relato'].head(3).to_string(index=False))

    print("\n📈 Gerando visualização 2D com PCA...")
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())

    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=celular_df['cluster'], cmap='tab10', s=30)
    plt.title(f"Clusters de relatos de roubo de celular (KMeans, K={ideal_k})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Salvar CSV para etapa de sumarização
    output_csv = f"cluster_{ideal_k}.csv"
    #celular_df[['relato_limpo']].to_csv(output_csv, index=False)
    celular_df[['relato_limpo', 'cluster']].to_csv(output_csv, index=False)
    print(f"\n💾 Arquivo salvo: {output_csv}")
    return output_csv
