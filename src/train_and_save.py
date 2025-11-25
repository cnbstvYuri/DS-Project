"""
Pipeline de Treinamento e Serialização de Modelos.

Este script é responsável por todo o ciclo de vida do treinamento:
1. Carregamento e Sanitização de Dados (Limpeza).
2. Correção de Labels (Target Inversion).
3. Engenharia de Atributos (via src.utils).
4. Definição de Pipelines de Pré-processamento (Imputação + Scaling).
5. Treinamento de Modelos (Random Forest e Logistic Regression).
6. Serialização (.joblib) para uso em produção.

Uso:
    python src/train_and_save.py --data data/heart.csv --target target
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, recall_score # Adicionado para validação final

# Importa a lógica centralizada de engenharia de features para garantir consistência
# com o ambiente de produção (app.py)

from utils import feature_engineering 

# Dicionário de mapeamento para padronizar nomes de colunas
COL_MAP = {
    'age':'age','sex':'sex','cp':'cp','trestbps':'resting_bp','chol':'chol','fbs':'fbs','restecg':'restecg',
    'thalach':'thalach','exang':'exang','oldpeak':'oldpeak','slope':'slope','ca':'ca','thal':'thal','target':'target'
}

def cap_outliers(df, cols, factor=3.0):
    """
    NOVA FUNÇÃO: Em vez de remover (o que perde pacientes graves), aplicamos um 'teto'.
    Valores muito acima de Q3 + 3*IQR são trazidos para o limite máximo aceitável.
    Isso mantém o dado do paciente doente, mas reduz o ruído estatístico.
    """
    df_capped = df.copy()
    print("🔧 Aplicando Capping em Outliers (preservando dados)...")
    
    for col in cols:
        if col not in df_capped.columns: continue
        
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definimos os limites (Teto e Piso)
        upper_limit = Q3 + (factor * IQR)
        lower_limit = Q1 - (factor * IQR)
        
        # .clip() força os valores a ficarem dentro desse intervalo
        df_capped[col] = df_capped[col].clip(lower=lower_limit, upper=upper_limit)
        
    return df_capped

def load_and_sanitize_data(path):
    """
    Carrega o dataset e aplica regras de qualidade de dados (Data Quality).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")
        
    print(f"📂 Carregando dataset: {path}")
    df = pd.read_csv(path)
    
    # 1. Padronização de Schema
    df = df.rename(columns={k:v for k,v in COL_MAP.items() if k in df.columns})
    
    # 2. Remoção de Duplicatas
    n_total = len(df)
    df = df.drop_duplicates()
    n_removidos = n_total - len(df)
    if n_removidos > 0:
        print(f"🧹 Data Cleaning: {n_removidos} linhas duplicadas removidas.")
    
    # 4. Tratamento Estatístico de Outliers (CORRIGIDO)
    # Anteriormente deletávamos linhas. Agora usamos Capping (Winsorization).
    # Isso impede que percamos pacientes com Oldpeak alto ou Pressão alta.
    cols_to_clean = ['chol', 'resting_bp', 'thalach', 'oldpeak']
    
    # Usamos fator 3.0 para ser bem conservador (só altera valores impossíveis/extremos)
    df = cap_outliers(df, cols_to_clean, factor=3.0)

    return df

def get_feature_lists(X: pd.DataFrame):
    """
    Separa automaticamente colunas numéricas e categóricas baseada em heurísticas.
    Colunas numéricas com baixa cardinalidade (<10 valores únicos) são tratadas como categóricas.
    """
    # Heurística: Numéricos com poucos valores únicos (ex: slope 0,1,2) viram categóricos
    potential_cat = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() < 10]
    
    # Numéricos puros
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    
    # Remove os "falsos numéricos" da lista
    for c in potential_cat:
        if c in num_cols:
            num_cols.remove(c)
            
    # Categóricos finais
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist() + potential_cat
    cat_cols = list(dict.fromkeys(cat_cols)) # Deduplicação
    
    # Garante integridade
    num_cols = [c for c in num_cols if c not in cat_cols]
    
    return num_cols, cat_cols

def main(args):
    # 1. ETL Inicial
    df = load_and_sanitize_data(args.data)
    
    # -----------------------------------------------------------
    # PADRONIZAÇÃO DO TARGET (Crucial para Interpretabilidade)
    # Análise exploratória indicou inversão no dataset original (0=Doença).
    # Invertemos aqui para garantir que 1=Doença (Padrão Positivo).
    # -----------------------------------------------------------
    if 'target' in df.columns:
        print("🔄 Normalização: Ajustando Target para padrão (1 = Doença Detectada)...")
        df['target'] = df['target'].apply(lambda x: 1 if x == 0 else 0)
        print(f"   Distribuição de Classes: {df['target'].value_counts().to_dict()}")
    
    # 2. Feature Engineering
    # Utiliza a função centralizada do utils para manter paridade com o Dashboard
    df = feature_engineering(df)
    
    # Validação de Colunas
    TARGET = args.target
    if TARGET not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET}' não encontrada no dataset.")
    
    # Seleção de Features (White-list)
    # Garante que apenas colunas conhecidas entrem no modelo
    features_list = [
        'age','sex','cp','resting_bp','chol','fbs','restecg','thalach','exang',
        'oldpeak','slope','ca','thal','AgeGroup','CholCategory','AgeOver50','CSI','RiskFactorsCount'
    ]
    features_final = [f for f in features_list if f in df.columns]
    
    X = df[features_final].copy()
    y = df[TARGET].copy()
    
    num_cols, cat_cols = get_feature_lists(X)
    print(f'⚙️ Setup: {len(num_cols)} features numéricas | {len(cat_cols)} features categóricas')
    
    # 3. Definição do Pré-processamento (Pipeline Robusto)
    
    # Pipeline Numérico: Imputação pela Mediana (para robustez a nulos) + Normalização Z-Score
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline Categórico: Imputação de Constante + OneHotEncoding
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # 4. Definição dos Modelos
    
    # Random Forest: Robusto para não-linearidades e interações
    # ADICIONADO: oob_score=True para avaliação interna em datasets pequenos
    rf = Pipeline([
        ('preproc', preprocessor), 
        ('model', RandomForestClassifier(
            n_estimators=args.n_estimators, 
            random_state=42, 
            max_depth=args.rf_max_depth, 
            min_samples_split=args.rf_min_samples_split, 
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight='balanced_subsample', # Mais agressivo que 'balanced'
            oob_score=True 
        ))
    ])
    
    # Regressão Logística: Baseline interpretável e probabilística
    lr = Pipeline([
        ('preproc', preprocessor), 
        ('model', LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced'))
    ])
    
    # 5. Split de Dados
    # Stratify=y garante que a proporção de doentes seja igual no treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    
    # 6. Treinamento
    print('🚀 Iniciando treinamento dos modelos...')
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # 7. Serialização (Salvamento)
    os.makedirs('models', exist_ok=True)
    
    # Salvamos os modelos treinados
    joblib.dump(rf, 'models/rf_model.joblib')
    joblib.dump(lr, 'models/lr_model.joblib')
    # Salvamos o preprocessor separadamente (útil para SHAP/Explainer)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    # Salvamos o conjunto de teste para avaliação honesta no Dashboard
    X_test.to_csv('models/X_test.csv', index=False)
    pd.Series(y_test).to_csv('models/y_test.csv', index=False, header=['target'])
    
    print('✅ Pipeline concluído. Artefatos salvos no diretório models/.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento Heart Disease Prediction")
    parser.add_argument('--data', type=str, required=True, help='Caminho para o CSV do dataset')
    parser.add_argument('--target', type=str, default='target', help='Nome da coluna alvo')
    parser.add_argument('--test_size', type=float, default=0.3, help='Proporção do conjunto de teste')
    
    # Hiperparâmetros RF - Ajustados para evitar Overfitting em dados pequenos
    parser.add_argument('--n_estimators', type=int, default=500) # Aumentado para estabilidade
    parser.add_argument('--rf_max_depth', type=int, default=5)   # Mantido baixo
    parser.add_argument('--rf_min_samples_split', type=int, default=10) # Aumentado para evitar nós muito específicos
    parser.add_argument('--rf_min_samples_leaf', type=int, default=5)   # Aumentado para garantir robustez
    
    args = parser.parse_args()
    main(args)