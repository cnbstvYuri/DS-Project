"""
Dashboard Interativo de Previsão de Doença Cardíaca.
Autor: Yuri Vaz Claro e Vinicius Boeira
Data: Novembro/2025
Descrição: Interface Streamlit para exploração de dados, validação de hipóteses e inferência de modelos ML.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
from src.utils import load_test_split, compute_metrics, get_feature_names_from_pipeline, feature_engineering
from src.explainability import compute_shap

# Configuração inicial da página (Layout Wide para melhor visualização de gráficos)
st.set_page_config(page_title="Dashboard Doença Cardíaca", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# 1. CONSTANTES E MAPEAMENTOS
# ==============================================================================
# Dicionários usados para traduzir códigos numéricos (0, 1, 2) para texto legível nos gráficos.

VAL_MAPS = {
    'sex': {0: 'Mulher', 1: 'Homem'},
    'cp': { 
        0: 'Angina Típica (Risco Baixo)', 
        1: 'Angina Atípica (Risco Alto)', 
        2: 'Dor Não-Anginosa (Risco Alto)', 
        3: 'Assintomático'
    },
    'fbs': {0: 'Glicemia < 120', 1: 'Glicemia > 120'}, 
    'exang': {0: 'Não', 1: 'Sim'}, 
    'slope': {
        0: 'Subindo (Upsloping)', 
        1: 'Plano (Flat)', 
        2: 'Descendo (Downsloping - Risco Alto)'
    }, 
    'thal': { 
        0: 'Nulo',
        1: 'Defeito Fixo', 
        2: 'Defeito Reversível (Alto Risco)', 
        3: 'Normal' 
    },
    'target': {0: 'Saudável', 1: 'Doença Detectada'},
    'AgeGroup': {'Young': 'Jovem', 'Adult': 'Adulto', 'Senior': 'Sênior', 'Elderly': 'Idoso'},
    'CholCategory': {'Desirable': 'Desejável', 'Borderline': 'Limítrofe', 'High': 'Alto'}
}

# Labels amigáveis para os eixos dos gráficos
LABEL_MAP = {
    'age': 'Idade', 'sex': 'Sexo', 'cp': 'Tipo de Dor no Peito', 'resting_bp': 'Pressão Arterial (Repouso)',
    'chol': 'Colesterol', 'fbs': 'Açúcar em Jejum', 'restecg': 'Eletrocardiograma',
    'thalach': 'Freq. Cardíaca Máx.', 'exang': 'Angina (Exercício)', 'oldpeak': 'Depressão ST',
    'slope': 'Inclinação ST', 'ca': 'Vasos Principais (0-3)', 'thal': 'Teste Tálio (Thal)', 'target': 'Diagnóstico',
    'AgeGroup': 'Faixa Etária', 'CholCategory': 'Categoria Colesterol'
}

# Mapeamento técnico para garantir consistência com o CSV original
COL_MAP = {
    'age':'age','sex':'sex','cp':'cp','trestbps':'resting_bp','chol':'chol','fbs':'fbs','restecg':'restecg',
    'thalach':'thalach','exang':'exang','oldpeak':'oldpeak','slope':'slope','ca':'ca','thal':'thal','target':'target'
}

DATA_PATH = 'data/heart.csv'

# ==============================================================================
# 2. FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO
# ==============================================================================

@st.cache_data
def load_data(path):
    """
    Carrega o dataset, renomeia colunas, remove duplicatas e aplica engenharia de atributos.
    O cache é usado para evitar recarregamento a cada interação do usuário.
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    
    # Padronização dos nomes das colunas
    df = df.rename(columns={k:v for k,v in COL_MAP.items() if k in df.columns})
    
    # Limpeza de dados: Remoção de duplicatas para evitar viés estatístico
    df = df.drop_duplicates()
    
    # Filtro de sanidade: Remove outliers impossíveis (erros de digitação)
    if 'oldpeak' in df.columns:
        df = df[df['oldpeak'] <= 20]

    # Engenharia de Atributos (Deve replicar a lógica usada no treinamento)
    # 1. Categorização de Idade e Colesterol
    df['AgeGroup'] = pd.cut(df['age'], bins=[0,39,54,69,120], labels=['Young','Adult','Senior','Elderly'])
    df['CholCategory'] = pd.cut(df['chol'], bins=[0,199,239,10000], labels=['Desirable','Borderline','High'])
    
    # 2. Binários e Índices Calculados
    df['AgeOver50'] = (df['age'] > 50).astype(int)
    
    # CSI (Cardiac Shock Index) - Trata divisão por zero
    df['CSI'] = df['resting_bp'] / df['thalach'].replace(0, np.nan)
    df['CSI'] = df['CSI'].fillna(df['CSI'].median())
    
    # Contagem de Fatores de Risco
    df['RiskFactorsCount'] = ((df['chol'] >= 240).astype(int) + 
                              (df['resting_bp'] >= 130).astype(int) + 
                              (df['age'] > 50).astype(int))

    return df

def get_visual_dataframe(df_raw):
    """
    Gera uma cópia do dataframe com valores traduzidos (ex: 0 -> 'Mulher').
    Utilizado exclusivamente para visualização em gráficos (Plotly).
    """
    if df_raw is None: return None
    df_vis = df_raw.copy()
    
    # Aplica os dicionários de tradução (VAL_MAPS)
    for col, mapping in VAL_MAPS.items():
        if col in df_vis.columns:
            # Converte para object/string para permitir textos nas colunas numéricas
            df_vis[col] = df_vis[col].astype(object)
            df_vis[col] = df_vis[col].map(mapping).fillna(df_vis[col])
            df_vis[col] = df_vis[col].astype(str)

    # Renomeia colunas para ficar amigável no mouse-over dos gráficos
    df_vis = df_vis.rename(columns=LABEL_MAP)
    return df_vis

@st.cache_resource
def load_models_if_exist():
    """Carrega os modelos treinados (.joblib) da pasta models/."""
    models = {}
    if os.path.exists('models/rf_model.joblib'):
        models['RandomForest'] = joblib.load('models/rf_model.joblib')
    if os.path.exists('models/lr_model.joblib'):
        models['LogisticRegression'] = joblib.load('models/lr_model.joblib')
    return models

# ==============================================================================
# 3. BARRA LATERAL (CONFIGURAÇÃO E FILTROS)
# ==============================================================================

st.sidebar.title("Navegação")

# Carregamento inicial
models = load_models_if_exist()
df = load_data(DATA_PATH)

# Menu de Navegação
menu_options = {
    "Visão Geral": "Overview",
    "Análise Exploratória (EDA)": "EDA",
    "Validação de Hipóteses": "Hypothesis",
    "Performance do Modelo": "Model Performance",
    "Explicabilidade (SHAP)": "Explainability",
    "Simulador (Previsão)": "Predict"
}
page_selection = st.sidebar.radio("Ir para:", list(menu_options.keys()))
page = menu_options[page_selection]

# Seleção de Modelo (Aparece apenas se existirem modelos carregados)
if models:
    model_choice = st.sidebar.selectbox("Modelo para Inferência", options=list(models.keys()))
else:
    model_choice = None

# Tratamento caso o dataset não exista
if df is None:
    st.sidebar.error("⚠️ Arquivo 'data/heart.csv' não encontrado.")
    df_visual = None
    df_filtered = None
else:
    # Prepara dados para visualização
    df_visual = get_visual_dataframe(df)
    df_filtered = df_visual.copy()

    # --- Filtros Dinâmicos ---
    st.sidebar.divider()
    st.sidebar.header("🔍 Filtros de Análise")
    
    # Filtro 1: Diagnóstico
    if 'Diagnóstico' in df_filtered.columns:
        all_diag = df_filtered['Diagnóstico'].unique()
        sel_diag = st.sidebar.multiselect("Diagnóstico:", all_diag, default=all_diag)
    else: sel_diag = []

    # Filtro 2: Tipo de Dor
    if 'Tipo de Dor no Peito' in df_filtered.columns:
        all_cp = df_filtered['Tipo de Dor no Peito'].unique()
        sel_cp = st.sidebar.multiselect("Tipo de Dor:", all_cp, default=all_cp)
    else: sel_cp = []

    # Filtro 3: Slider de Colesterol
    if 'chol' in df.columns:
        min_chol = int(df['chol'].min())
        max_chol = int(df['chol'].max())
        range_chol = st.sidebar.slider("Faixa de Colesterol:", min_chol, max_chol, (min_chol, max_chol))
    else: range_chol = (0, 1000)
    
    # Aplicação dos Filtros (Lógica de Máscaras)
    if sel_diag:
        df_filtered = df_filtered[df_filtered['Diagnóstico'].isin(sel_diag)]
    
    if sel_cp:
        df_filtered = df_filtered[df_filtered['Tipo de Dor no Peito'].isin(sel_cp)]
        
    # Filtro numérico cruzando o índice do DF original com o DF visual
    mask_chol = (df['chol'] >= range_chol[0]) & (df['chol'] <= range_chol[1])
    df_filtered = df_filtered[df_filtered.index.isin(df[mask_chol].index)]

    st.sidebar.caption(f"Pacientes filtrados: {len(df_filtered)}")

# ==============================================================================
# 4. PÁGINAS DO DASHBOARD
# ==============================================================================

# --- PÁGINA: VISÃO GERAL ---
if page == 'Overview':
    st.header("Visão Geral do Dataset")
    
    if df_filtered is None or df_filtered.empty:
        st.warning("Nenhum dado disponível com os filtros atuais.")
    else:
        st.subheader("Amostra dos Dados")
        st.dataframe(df_filtered.head())
        
        st.subheader("Distribuição do Alvo (Target)")
        target_col = LABEL_MAP.get('target', 'Diagnóstico')
        
        fig = px.histogram(df_filtered, x=target_col, color=target_col, 
                           title='Balanceamento das Classes (Saudável vs Doença)', text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("⚠️ Limitações e Trabalhos Futuros")
        st.markdown("""
        Embora o modelo apresente excelente performance (AUC > 0.90), identificamos pontos de melhoria:
        1.  **Otimização:** Implementar `GridSearchCV` para refinar hiperparâmetros.
        2.  **Validação:** Aplicar K-Fold Cross-Validation para maior robustez estatística.
        3.  **Dados:** A coleta de mais dados reais é essencial para generalização.
        """)

# --- PÁGINA: ANÁLISE EXPLORATÓRIA (EDA) ---
if page == 'EDA':
    st.header("Análise Exploratória de Dados")
    
    if df_filtered is None or df_filtered.empty:
        st.warning("Dados insuficientes para gerar gráficos.")
    else:
        target_col = LABEL_MAP.get('target', 'Diagnóstico')

        st.subheader("Correlação com o Diagnóstico (O que mais impacta?)")
        st.markdown("Este gráfico mostra quais variáveis têm maior relação matemática com a doença.")
        
        # Prepara os dados numéricos
        df_numeric = df.select_dtypes(include=[np.number])
        
        # Calcula correlação apenas com o TARGET
        corr_target = df_numeric.corrwith(df_numeric['target']).sort_values(ascending=False)
        
        # Remove o próprio target da lista (que seria 1.0)
        corr_target = corr_target.drop('target', errors='ignore')
        
        # Cria um gráfico de barras horizontal colorido
        fig_corr = px.bar(
            x=corr_target.values,
            y=corr_target.index,
            orientation='h',
            title="Correlação de Pearson com a Doença Cardíaca",
            labels={'x': 'Força da Correlação (-1 a 1)', 'y': 'Variável'},
            color=corr_target.values,
            color_continuous_scale='RdBu_r', # Vermelho = Positivo (Risco), Azul = Negativo (Proteção)
            range_color=[-1, 1]
        )
        # Adiciona uma linha vertical no zero
        fig_corr.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
        
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Barras para a **Direita (Vermelho)** aumentam o risco. Barras para a **Esquerda (Azul)** diminuem o risco.")
        # Recupera dados numéricos correspondentes ao filtro visual atual
        df_numeric_filtered = df.loc[df_filtered.index]
        corr = df_numeric_filtered.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

        # Gráficos Categóricos e Distribuições
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Perfil por Sexo")
            sex_col = LABEL_MAP.get('sex', 'Sexo')
            fig = px.histogram(df_filtered, x=sex_col, color=target_col, barmode='group', text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Tipo de Dor no Peito")
            cp_col = LABEL_MAP.get('cp', 'Tipo de Dor no Peito')
            fig = px.histogram(df_filtered, x=cp_col, color=target_col, barmode='group')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Distribuição de Idade por Grupo")
        age_col = LABEL_MAP.get('age', 'Idade')
        fig = px.box(df_filtered, x=target_col, y=age_col, color=target_col, points="all")
        st.plotly_chart(fig, use_container_width=True)
        
        thal_col = LABEL_MAP.get('thal', 'Teste Tálio (Thal)')
        if thal_col in df_filtered.columns:
            st.markdown("### Resultado do Teste de Tálio")
            fig = px.histogram(df_filtered, x=thal_col, color=target_col, barmode='group')
            st.plotly_chart(fig, use_container_width=True)

# --- PÁGINA: TESTE DE HIPÓTESES ---
if page == 'Hypothesis':
    st.header("🧪 Validação de Hipóteses Clínicas")
    st.markdown("Validação estatística de premissas médicas baseada nos dados coletados.")

    if df_visual is not None:
        target_col = LABEL_MAP.get('target', 'Diagnóstico')
        age_col = LABEL_MAP.get('age', 'Idade')
        exang_col = LABEL_MAP.get('exang', 'Angina (Exercício)')
        thalach_col = LABEL_MAP.get('thalach', 'Freq. Cardíaca Máx.')

        # H1: Idade
        st.divider()
        st.subheader("H1 — Pacientes mais velhos têm maior probabilidade de doença?")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_h1 = px.box(df_visual, x=target_col, y=age_col, color=target_col, 
                            color_discrete_map={'Saudável': 'blue', 'Doença Detectada': 'red'})
            st.plotly_chart(fig_h1, use_container_width=True)
        with c2:
            media_saudavel = df[df['target'] == 0]['age'].mean()
            media_doente = df[df['target'] == 1]['age'].mean()
            st.metric("Média Idade (Doentes)", f"{media_doente:.1f}", delta=f"{media_doente - media_saudavel:.1f}")
            if media_doente > media_saudavel:
                st.success("✅ **CONFIRMADA**")
            else:
                st.warning("⚠️ **INCONCLUSIVA**")

        # H2: Angina
        st.divider()
        st.subheader("H2 — Angina no exercício indica maior risco?")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_h2 = px.histogram(df_visual, x=exang_col, color=target_col, barmode='group', text_auto=True,
                                  color_discrete_map={'Saudável': 'blue', 'Doença Detectada': 'red'})
            st.plotly_chart(fig_h2, use_container_width=True)
        with c2:
            # Cálculo de proporção de risco
            total_angina = len(df[df['exang'] == 1])
            doentes_angina = len(df[(df['exang'] == 1) & (df['target'] == 1)])
            perc = (doentes_angina / total_angina * 100) if total_angina > 0 else 0
            st.metric("Risco com Angina", f"{perc:.1f}%")
            if perc > 50: st.success("✅ **CONFIRMADA**")
            else: st.error("❌ **REFUTADA**")

        # H3: Frequência Cardíaca
        st.divider()
        st.subheader("H3 — Frequência cardíaca baixa indica doença?")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_h3 = px.box(df_visual, x=target_col, y=thalach_col, color=target_col,
                            color_discrete_map={'Saudável': 'blue', 'Doença Detectada': 'red'})
            st.plotly_chart(fig_h3, use_container_width=True)
        with c2:
            media_fc_saudavel = df[df['target'] == 0]['thalach'].mean()
            media_fc_doente = df[df['target'] == 1]['thalach'].mean()
            st.metric("BPM Médio (Doentes)", f"{media_fc_doente:.0f}", delta=f"{media_fc_doente - media_fc_saudavel:.0f}")
            if media_fc_doente < media_fc_saudavel: st.success("✅ **CONFIRMADA**")
            else: st.warning("⚠️ **REFUTADA**")

# --- PÁGINA: PERFORMANCE DO MODELO ---
if page == 'Model Performance':
    st.header("Avaliação de Performance (Test Set)")
    
    # Carrega dados de teste (nunca vistos pelo modelo)
    X_test, y_test = load_test_split()
    
    if X_test is None:
        st.warning("Dataset de teste não encontrado. Execute o treinamento primeiro.")
    else:
        models_dict = load_models_if_exist()
        for name, model in models_dict.items():
            st.subheader(f"Modelo: {name}")
            metrics = compute_metrics(model, X_test, y_test)
            
            # Exibição de KPIs
            c1, c2, c3 = st.columns(3)
            c1.metric("Acurácia", f"{metrics.get('accuracy',0):.1%}")
            if 'roc_auc' in metrics: 
                c2.metric("ROC AUC", f"{metrics.get('roc_auc',0):.3f}")
            
            # Matriz de Confusão
            st.markdown("**Matriz de Confusão**")
            cm = pd.DataFrame(metrics['confusion_matrix'], 
                              index=['Real: Saudável', 'Real: Doença'], 
                              columns=['Pred: Saudável', 'Pred: Doença'])
            st.dataframe(cm)

# --- PÁGINA: EXPLICABILIDADE (SHAP) ---
if page == 'Explainability':
    st.header("Explicabilidade do Modelo (SHAP)")
    st.info("Visualização das variáveis que mais impactam a decisão do modelo (Feature Importance Global).")
    
    X_test, y_test = load_test_split()
    models_dict = load_models_if_exist()
    
    if models_dict and X_test is not None:
        sel_model = st.selectbox("Escolha o Modelo", list(models_dict.keys()), key='shap_model')
        model = models_dict[sel_model]
        
        if st.button("Gerar Gráfico SHAP"):
            try:
                # Amostragem para performance
                sample = X_test.sample(min(100, len(X_test)), random_state=42)
                
                # Separação do Pipeline (Preprocessor vs Modelo) para compatibilidade com SHAP
                model_to_explain = model
                data_to_explain = sample

                if hasattr(model, 'named_steps'):
                    step_name = list(model.named_steps.keys())[0] 
                    preprocessor = model.named_steps[step_name]
                    data_to_explain = preprocessor.transform(sample)
                    
                    model_step_name = list(model.named_steps.keys())[-1]
                    model_to_explain = model.named_steps[model_step_name]
                    
                    if hasattr(data_to_explain, "toarray"):
                        data_to_explain = data_to_explain.toarray()
                    
                    # Tenta recuperar nomes das features para o gráfico
                    try:
                        feature_names = get_feature_names_from_pipeline(model, sample)
                        data_to_explain = pd.DataFrame(data_to_explain, columns=feature_names)
                    except: pass

                # Cálculo e Plotagem
                shap_values = compute_shap(model_to_explain, data_to_explain)
                
                # Ajuste para Random Forest (3 dimensões)
                if len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]
                
                import shap
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots()
                shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(plt.gcf())
                
            except Exception as e:
                st.error(f"Erro ao gerar SHAP: {e}")

# --- PÁGINA: SIMULADOR (PREDICT) ---
if page == 'Predict':
    st.header("Simulador de Risco Cardíaco")
    st.markdown("Preencha os dados clínicos para obter uma estimativa de risco em tempo real.")
    
    X_test, _ = load_test_split() # Usado apenas para pegar valores padrão (medianas)

    # Definição dos Inputs (Mapeamento Visual -> Numérico)
    sex_opts = {'Mulher': 0, 'Homem': 1}
    cp_opts = {'Angina Típica (0)': 0, 'Angina Atípica (1)': 1, 'Dor Não-Anginosa (2)': 2, 'Assintomático (3)': 3}
    fbs_opts = {'Normal (<120)': 0, 'Diabetes (>120)': 1}
    restecg_opts = {'Normal': 0, 'Anormalidade ST-T': 1, 'Hipertrofia Ventricular': 2}
    exang_opts = {'Não': 0, 'Sim': 1}
    slope_opts = {'Subindo (Melhor)': 0, 'Plano (Alerta)': 1, 'Descendo (Pior)': 2}
    thal_opts = {'Normal': 3, 'Defeito Fixo': 1, 'Defeito Reversível (Grave)': 2}

    with st.form("form_previsao"):
        st.subheader("Dados Clínicos")
        c1, c2, c3 = st.columns(3)
        with c1: age = st.number_input("Idade", 1, 120, 60)
        with c2: sex_label = st.selectbox("Sexo", list(sex_opts.keys()))
        with c3: cp_label = st.selectbox("Tipo de Dor", list(cp_opts.keys()))

        c4, c5, c6 = st.columns(3)
        with c4: resting_bp = st.number_input("Pressão Arterial", 50, 250, 120)
        with c5: chol = st.number_input("Colesterol", 100, 600, 200)
        with c6: fbs_label = st.selectbox("Glicemia", list(fbs_opts.keys()))

        st.subheader("Exames Cardíacos")
        c7, c8, c9 = st.columns(3)
        with c7: restecg_label = st.selectbox("ECG Repouso", list(restecg_opts.keys()))
        with c8: thalach = st.number_input("Freq. Máxima", 60, 220, 150)
        with c9: exang_label = st.selectbox("Angina (Exercício)", list(exang_opts.keys()))
        
        c10, c11, c12 = st.columns(3)
        with c10: oldpeak = st.number_input("Depressão ST", 0.0, 10.0, 0.0, step=0.1)
        with c11: slope_label = st.selectbox("Slope ST", list(slope_opts.keys()))
        with c12: ca = st.slider("Vasos Coloridos (0-3)", 0, 3, 0)
        
        thal_label = st.selectbox("Teste Tálio", list(thal_opts.keys()))
        submit = st.form_submit_button("CALCULAR RISCO")

    if submit:
        # 1. Montagem do Vetor de Entrada
        user_input = {
            'age': age, 'sex': sex_opts[sex_label], 'cp': cp_opts[cp_label],
            'resting_bp': resting_bp, 'chol': chol, 'fbs': fbs_opts[fbs_label],
            'restecg': restecg_opts[restecg_label], 'thalach': thalach,
            'exang': exang_opts[exang_label], 'oldpeak': oldpeak,
            'slope': slope_opts[slope_label], 'ca': ca, 'thal': thal_opts[thal_label]
        }
        X_new = pd.DataFrame([user_input])

        # 2. Engenharia de Features (Deve ser idêntica ao treinamento)
        # Recriação das variáveis sintéticas (AgeGroup, CSI, RiskFactors, etc.)
        X_new = feature_engineering(X_new)

        # 3. Inferência
        if model_choice in models:
            model = models[model_choice]
            try:
                # Reordena colunas para bater com o treino
                if hasattr(model, "feature_names_in_"):
                    X_new = X_new[model.feature_names_in_]
                
                pred = model.predict(X_new)[0]
                proba = model.predict_proba(X_new)[:,1][0] # Probabilidade da Classe 1 (Doença)

                st.divider()
                # 1 = Doença (Conforme treinamento corrigido)
                if pred == 1:
                    st.error("🚨 ALTO RISCO DETECTADO")
                    st.write(f"Probabilidade estimada: **{proba:.1%}**")
                    st.warning("Recomendação: Avaliação médica prioritária.")
                else:
                    st.success("✅ BAIXO RISCO DETECTADO")
                    st.write(f"Probabilidade de Doença: **{proba:.1%}**")
                    st.info("Mantenha o acompanhamento de rotina.")

            except Exception as e:
                st.error(f"Erro na predição: {e}")
        else:
            st.error("Modelo não carregado.")