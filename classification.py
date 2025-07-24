
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC

# Funções auxiliares

def preprocess(text):
    if pd.isnull(text):
        return ''
    text = str(text)
    text = re.sub(r'\W', ' ', text.lower())
    words = text.split()
    return ' '.join(words)

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

def classificar_e_avaliar(input_file):
    column_name = "especificacao_crime"
    result_df = search_and_label(input_file, column_name)
    if result_df is None or result_df.empty:
        print("Erro: os dados não foram carregados corretamente.")
        return
    print("\n✅ Dados carregados e rotulados com sucesso.")

    # Pré-processar texto
    result_df['especificacao_crime'] = result_df['especificacao_crime'].fillna('').apply(preprocess)
    result_df['relato'] = result_df['relato'].fillna('').apply(preprocess)
    result_df['combined_text'] = result_df['especificacao_crime'] + ' ' + result_df['relato']

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(result_df['combined_text']).toarray()
    y = result_df['label'].values

    # Balanceamento
    counts = Counter(y)
    minority_count = min(counts.values())

    balanced_indices = []
    np.random.seed(42)
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        selected_indices = np.random.choice(cls_indices, size=minority_count, replace=False)
        balanced_indices.extend(selected_indices)

    balanced_indices = np.array(balanced_indices)
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices].astype(int)

    balanced_df = pd.DataFrame(X_balanced, columns=vectorizer.get_feature_names_out())
    balanced_df['label'] = y_balanced

    # Avaliação com diferentes percentuais
    percentages = [100, 80, 60, 40, 20, 10]
    metrics_dict = {"SampleSize": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}

    for pct in percentages:
        sample_per_class = int(minority_count * (pct / 100.0))
        subset_list = []
        for cls in balanced_df['label'].unique():
            df_cls = balanced_df[balanced_df['label'] == cls]
            subset = df_cls.sample(n=min(len(df_cls), sample_per_class), random_state=42)
            subset_list.append(subset)

        subset_df = pd.concat(subset_list).sample(frac=1, random_state=42).reset_index(drop=True)
        X_subset = subset_df.drop(columns=['label']).values
        y_subset = subset_df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
        model = LinearSVC(random_state=42, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        metrics_dict["SampleSize"].append(sample_per_class)
        metrics_dict["Accuracy"].append(acc)
        metrics_dict["Precision"].append(prec)
        metrics_dict["Recall"].append(rec)
        metrics_dict["F1"].append(f1)

        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Transeunte', 'Celular', 'Moto', 'Coletivos', 'Resto'],
                    yticklabels=['Transeunte', 'Celular', 'Moto', 'Coletivos', 'Resto'])
        plt.xlabel('Previsão')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusão - {sample_per_class} amostras por classe')
        plt.show()

    metrics_df = pd.DataFrame(metrics_dict)
    print("\nResultados das Métricas:")
    print(metrics_df)

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_dict["SampleSize"], metrics_dict["Accuracy"], marker='o', label="Acurácia")
    plt.plot(metrics_dict["SampleSize"], metrics_dict["Precision"], marker='o', label="Precisão")
    plt.plot(metrics_dict["SampleSize"], metrics_dict["Recall"], marker='o', label="Recall")
    plt.plot(metrics_dict["SampleSize"], metrics_dict["F1"], marker='o', label="F1-Score")
    plt.xlabel("Número de amostras por classe")
    plt.ylabel("Valor da Métrica")
    plt.title("Impacto da Redução das Amostras na Performance do Modelo (5 classes)")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.show()