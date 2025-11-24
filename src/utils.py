"""
Módulo de Utilitários (Helpers) para o Projeto de Doença Cardíaca.

Este módulo centraliza funções críticas para garantir que o Treinamento e o Dashboard
compartilhem a mesma lógica de processamento de dados e avaliação.

Contém:
1. Engenharia de Atributos (Criação de variáveis)
2. Carregamento de Modelos e Dados
3. Cálculo de Métricas de Performance
4. Introspecção de Pipelines (Extração de nomes de colunas)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score, 
    confusion_matrix, classification_report
)

# ==============================================================================
# 1. ENGENHARIA DE ATRIBUTOS (Feature Engineering)
# ==============================================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica as transformações e criação de novas variáveis no dataset.
    
    IMPORTANTE: Esta função deve ser usada tanto no script de treinamento quanto
    no simulador (app.py) para garantir que o modelo receba os dados no mesmo formato.
    
    Args:
        df (pd.DataFrame): DataFrame com colunas originais (age, chol, etc.)
        
    Returns:
        pd.DataFrame: DataFrame enriquecido com novas colunas (AgeGroup, CSI, etc.)
    """
    df = df.copy()
    
    # 1. Categorização de Variáveis Contínuas (Binning)
    # Facilita para modelos baseados em árvore capturarem padrões não-lineares
    if 'age' in df.columns:
        df['AgeGroup'] = pd.cut(df['age'], bins=[0,39,54,69,120], labels=['Young','Adult','Senior','Elderly'])
        # Preenche nulos caso a idade esteja fora do range (improvável, mas seguro)
        df['AgeGroup'] = df['AgeGroup'].astype(object).fillna('Unknown')
        
        # Variável binária de risco idade
        df['AgeOver50'] = (df['age'] > 50).astype(int)

    if 'chol' in df.columns:
        df['CholCategory'] = pd.cut(df['chol'], bins=[0,199,239,10000], labels=['Desirable','Borderline','High'])
        df['CholCategory'] = df['CholCategory'].astype(object).fillna('Unknown')

    # 2. Criação de Indicadores Clínicos
    # CSI (Cardiac Shock Index): Razão entre Pressão e Frequência Cardíaca
    if 'resting_bp' in df.columns and 'thalach' in df.columns:
        # Evita divisão por zero substituindo 0 por NaN temporariamente
        df['CSI'] = df['resting_bp'] / df['thalach'].replace(0, np.nan)
        # Preenche falhas com a mediana (valor neutro)
        df['CSI'] = df['CSI'].fillna(df['CSI'].median())

    # 3. Contagem de Fatores de Risco (Risk Score Simplificado)
    # Soma +1 para cada condição de risco atingida
    required_cols = ['chol', 'resting_bp', 'age']
    if all(col in df.columns for col in required_cols):
        df['RiskFactorsCount'] = (
            (df['chol'] >= 240).astype(int) +
            (df['resting_bp'] >= 130).astype(int) +
            (df['age'] > 50).astype(int)
        )
        
    return df

# ==============================================================================
# 2. IO & CARREGAMENTO
# ==============================================================================

def load_model(path):
    """Carrega um objeto serializado (modelo ou pipeline) via joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")
    return joblib.load(path)

def load_test_split(models_dir='models'):
    """
    Carrega o conjunto de teste (X_test, y_test) salvo durante o treinamento.
    Essencial para garantir que as métricas do dashboard sejam honestas (dados nunca vistos).
    """
    x_path = os.path.join(models_dir, 'X_test.csv')
    y_path = os.path.join(models_dir, 'y_test.csv')
    
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None
        
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).iloc[:, 0] # Garante que seja uma Series, não DataFrame
    
    return X_test, y_test

# ==============================================================================
# 3. MÉTICAS DE AVALIAÇÃO
# ==============================================================================

def compute_metrics(model, X, y):
    """
    Calcula métricas completas de classificação para o Dashboard.
    Retorna um dicionário pronto para consumo visual.
    """
    # Predições "Hard" (0 ou 1)
    y_pred = model.predict(X)
    
    # Predições "Soft" (Probabilidades) - Necessário para ROC/AUC
    y_proba = None
    try:
        y_proba = model.predict_proba(X)[:, 1] # Pega probabilidade da classe positiva (1)
    except Exception:
        # Alguns modelos (como SVM sem flag de probabilidade) podem falhar aqui
        pass

    results = {}
    
    # Métricas Básicas
    results['accuracy'] = accuracy_score(y, y_pred)
    results['classification_report'] = classification_report(y, y_pred, output_dict=True)
    results['confusion_matrix'] = confusion_matrix(y, y_pred).tolist() # .tolist() para compatibilidade JSON/UI

    # Métricas Avançadas (Threshold-Independent)
    if y_proba is not None:
        try:
            results['roc_auc'] = roc_auc_score(y, y_proba)
            
            # Dados para plotar Curva ROC
            fpr, tpr, _ = roc_curve(y, y_proba)
            results['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            
            # Dados para plotar Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y, y_proba)
            results['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
            
            results['average_precision'] = average_precision_score(y, y_proba)
        except Exception:
            # Evita quebrar o dashboard se houver erro matemático em casos de borda
            pass
            
    return results

# ==============================================================================
# 4. INTROSPECÇÃO DE MODELOS
# ==============================================================================

def get_feature_names_from_pipeline(pipeline, X_sample=None):
    """
    Tenta extrair os nomes das colunas após o pré-processamento (OneHotEncoding, etc.).
    Útil para mostrar Feature Importance com nomes reais em vez de índices genéricos.
    """
    try:
        # Tenta localizar o passo de pré-processamento no Pipeline
        # Nomes comuns: 'preproc', 'preprocess', 'preprocessor'
        pre = (pipeline.named_steps.get('preproc') or 
               pipeline.named_steps.get('preprocess') or 
               pipeline.named_steps.get('preprocessor'))
        
        if pre is None:
            # Se não achar, devolve as colunas originais como fallback
            return X_sample.columns.tolist() if X_sample is not None else None
            
        try:
            # Método padrão do Scikit-Learn moderno
            return pre.get_feature_names_out()
        except Exception:
            # Fallback para versões antigas ou transformers customizados
            return X_sample.columns.tolist() if X_sample is not None else None
            
    except Exception:
        return None