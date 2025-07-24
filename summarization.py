
import os
import pandas as pd
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

def extract_from_text_list(texts):
    docs = [Document(page_content=t) for t in texts if isinstance(t, str) and t.strip() != ""]
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    return splitter.split_documents(docs)

def summarize_with_kmeans(
    docs,
    llm,
    embeddings,
    summary_prompt: str = None,
    num_clusters: int = 8
) -> str:
    cluster_filter = EmbeddingsClusteringFilter(
        embeddings   = embeddings,
        num_clusters = num_clusters
    )
    clustered = cluster_filter.transform_documents(documents=docs)

    if summary_prompt:
        prompt = PromptTemplate(
            input_variables=["text"],
            template=summary_prompt
        )
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    else:
        chain = load_summarize_chain(llm, chain_type="stuff")

    return chain.invoke(clustered)

def resumir_clusters(csv_path: str) -> str:
    print(f"\n📚 Iniciando sumarização com base no arquivo: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'relato_limpo' not in df.columns or 'cluster' not in df.columns:
        raise ValueError("O CSV precisa conter as colunas 'relato_limpo' e 'cluster'.")

    embeddings = HuggingFaceEmbeddings(
        model_name    = "BAAI/bge-base-en-v1.5",
        model_kwargs  = {"device": "cuda"},
        encode_kwargs = {"normalize_embeddings": True},
    )

    llm = ChatOllama(model="llama3.1:latest", temperature=0)

    custom_prompt = (
        """
        A seguir você terá vários relatos referentes a crimes de roubo. Faça uma análise geral e resuma o conteúdo 
        de uma forma que identifique o modus operandi mais comum nesses relatos. É necessário manter o contexto 
        e somente resumir fatos verdadeiros contidos nos relatos: *resuma em no mínimo 200 tokens*:

        {text}

        Resumo:
        """
    )

    resumos = []
    for cluster_id in sorted(df['cluster'].unique()):
        relatos_cluster = df[df['cluster'] == cluster_id]['relato_limpo'].dropna().astype(str).tolist()
        print(f"\n🔹 Gerando resumo para Cluster {cluster_id} com {len(relatos_cluster)} relatos...")

        if not relatos_cluster:
            resumos.append(f"📂 Cluster {cluster_id}: (Sem relatos disponíveis)")
            continue

        docs = extract_from_text_list(relatos_cluster)
        resumo = summarize_with_kmeans(
            docs=docs,
            llm=llm,
            embeddings=embeddings,
            summary_prompt=custom_prompt,
            num_clusters=5
        )
        resumos.append(f"📂 Cluster {cluster_id}: {resumo}")

    return "\n\n".join(resumos)
