import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings
import re

# Configurazione Pagina
st.set_page_config(page_title="⚽ Football Analytics Pro", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Titolo
st.title("⚽ Dashboard Analisi Calcio Pro")
st.markdown("**Statistiche Gol, Previsioni Poisson & Analisi Ritmo (Kaplan-Meier)**")

# ==========================================
# 1. CARICAMENTO DATI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📂 Caricamento Dati")
    uploaded_file = st.file_uploader("Carica il file CSV/Excel", type=['csv', 'xlsx'])

if uploaded_file is None:
    st.info("👈 Carica un file dal menu a sinistra per iniziare.")
    st.stop()

# Funzione Caricamento Robusto
@st.cache_data
def load_data(file):
    try:
        # Tenta lettura CSV con separatore automatico
        try:
            # Legge prima riga per capire separatore
            line = file.readline().decode('latin1')
            file.seek(0) # Torna all'inizio
            sep = ';' if line.count(';') > line.count(',') else ','
            
            df = pd.read_csv(file, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
        except:
            # Prova come Excel
            df = pd.read_excel(file)

        # Pulizia Colonne
        df.columns = df.columns.astype(str).str.strip().str.upper()
        df = df.loc[:, ~df.columns.duplicated()]

        # Mappatura
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY'],
            'CASA': ['CASA', 'HOME', 'TEAM1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TEAM2']
        }
        
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    if candidate in df.columns:
                        df.rename(columns={candidate: target}, inplace=True)
                        break
        
        # Pulizia celle
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # Filtro Campionato
        if 'PAESE' in df.columns:
            df['ID_LEGA'] = df['PAESE'] + " - " + df['LEGA']
        else:
            df['ID_LEGA'] = df['LEGA']
            
        return df

    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
        return pd.DataFrame()

df = load_data(uploaded_file)

if df.empty:
    st.error("Il file caricato è vuoto o non valido.")
    st.stop()

st.success(f"✅ File caricato: {len(df)} partite analizzate.")

# ==========================================
# 2. SELEZIONE DATI
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    leghe = sorted(df['ID_LEGA'].unique())
    sel_lega = st.selectbox("🏆 Seleziona Campionato", leghe)

df_league = df[df['ID_LEGA'] == sel_lega].copy()
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())

with col2:
    sel_home = st.selectbox("🏠 Squadra Casa", teams, index=0)

with col3:
    # Cerca di selezionare una squadra diversa di default
    idx_away = 1 if len(teams) > 1 else 0
    sel_away = st.selectbox("✈️ Squadra Ospite", teams, index=idx_away)

# ==========================================
# 3. ENGINE DI ANALISI
# ==========================================
if st.button("🚀 AVVIA ANALISI MATCH"):
    st.divider()
    st.subheader(f"⚔️ Match Analysis: {sel_home} vs {sel_away}")
    
    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def get_minutes(val):
        if pd.isna(val): return []
        s = str(val).replace(',', '.').replace(';', ' ')
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        res = []
        for x in nums:
            try:
                n = int(float(x))
                if 0 <= n <= 130: res.append(n)
            except: pass
        return res

    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else 'GOALMINCASA'
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else 'GOALMINOSPITE'

    # Accumulatori
    goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
    match_h, match_a = 0, 0
    times_h, times_a, times_league = [], [], []
    
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # Dati Lega (per Media)
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # Heatmap
        if h in stats_match:
            for m in min_h:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[h]['F'][intervals[idx]] += 1
            for m in min_a:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[h]['S'][intervals[idx]] += 1
        
        if a in stats_match:
            for m in min_a:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[a]['F'][intervals[idx]] += 1
            for m in min_h:
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[a]['S'][intervals[idx]] += 1

        # Medie
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(min_h)
            goals_h['HT'] += len([x for x in min_h if x <= 45])
            goals_h['S_FT'] += len(min_a)
            goals_h['S_HT'] += len([x for x in min_a if x <= 45])
            if min_h: times_h.append(min(min_h))
        
        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(min_a)
            goals_a['HT'] += len([x for x in min_a if x <= 45])
            goals_a['S_FT'] += len(min_h)
            goals_a['S_HT'] += len([x for x in min_h if x <= 45])
            if min_a: times_a.append(min(min_a))

    # Calcoli Medie
    avg_h_ft = goals_h['FT'] / match_h if match_h else 0
    avg_h_ht = goals_h['HT'] / match_h if match_h else 0
    avg_h_conc_ft = goals_h['S_FT'] / match_h if match_h else 0
    avg_h_conc_ht = goals_h['S_HT'] / match_h if match_h else 0

    avg_a_ft = goals_a['FT'] / match_a if match_a else 0
    avg_a_ht = goals_a['HT'] / match_a if match_a else 0
    avg_a_conc_ft = goals_a['S_FT'] / match_a if match_a else 0
    avg_a_conc_ht = goals_a['S_HT'] / match_a if match_a else 0

    # Poisson
    exp_h_ft = (avg_h_ft + avg_a_conc_ft) / 2
    exp_a_ft = (avg_a_ft + avg_h_conc_ft) / 2
    exp_h_ht = (avg_h_ht + avg_a_conc_ht) / 2
    exp_a_ht = (avg_a_ht + avg_h_conc_ht) / 2

    def calc_poisson(lam_h, lam_a):
        probs = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
        return np.sum(np.tril(probs, -1)), np.sum(np.diag(probs)), np.sum(np.triu(probs, 1))

    p1_ft, px_ft, p2_ft = calc_poisson(exp_h_ft, exp_a_ft)
    
    prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_u15_ht = prob_00_ht + (poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)) + (poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht))

    # --- VISUALIZZAZIONE RISULTATI ---
    
    # 1. Statistiche Medie
    st.write("### 📊 Medie Gol")
    col_stats1, col_stats2 = st.columns(2)
    
    with col_stats1:
        st.info(f"**🏠 {sel_home} (Casa)**\n\n"
                f"**1° Tempo:** {avg_h_ht:.2f} Fatti | {avg_h_conc_ht:.2f} Subiti\n\n"
                f"**Finale:** {avg_h_ft:.2f} Fatti | {avg_h_conc_ft:.2f} Subiti")
    
    with col_stats2:
        st.warning(f"**✈️ {sel_away} (Ospite)**\n\n"
                 f"**1° Tempo:** {avg_a_ht:.2f} Fatti | {avg_a_conc_ht:.2f} Subiti\n\n"
                 f"**Finale:** {avg_a_ft:.2f} Fatti | {avg_a_conc_ft:.2f} Subiti")

    # 2. Previsioni Poisson
    st.write("### 🎲 Previsioni Matematiche (Poisson)")
    col_pois1, col_pois2, col_pois3 = st.columns(3)
    
    col_pois1.metric("1 (Casa)", f"{p1_ft*100:.1f}%")
    col_pois2.metric("X (Pareggio)", f"{px_ft*100:.1f}%")
    col_pois3.metric("2 (Ospite)", f"{p2_ft*100:.1f}%")
    
    st.write(f"**Analisi 1° Tempo:** 0-0 ({prob_00_ht*100:.1f}%) | Under 1.5 ({prob_u15_ht*100:.1f}%)")

    # 3. Grafici
    tab1, tab2 = st.tabs(["📉 Ritmo Gol (Kaplan-Meier)", "🔥 Heatmap Densità"])

    with tab1:
        if times_h and times_a:
            fig, ax = plt.subplots(figsize=(10, 5))
            kmf_h = KaplanMeierFitter()
            kmf_a = KaplanMeierFitter()
            kmf_l = KaplanMeierFitter()
            
            kmf_h.fit(times_h, label=f'{sel_home} (1° Gol)')
            kmf_a.fit(times_a, label=f'{sel_away} (1° Gol)')
            if times_league:
                kmf_l.fit(times_league, label='Media Campionato')
                kmf_l.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')
            
            kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            # Mediana
            med_h = kmf_h.median_survival_time_
            med_a = kmf_a.median_survival_time_
            plt.axhline(y=0.5, color='green', linestyle=':', label='Mediana (50%)')
            
            plt.title('Tempo al 1° Gol')
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig)
            st.caption(f"Minuto mediano 1° gol: {sel_home} ~{med_h:.0f}' | {sel_away} ~{med_a:.0f}'")
        else:
            st.warning("Dati insufficienti per il grafico KM.")

    with tab2:
        rows_f = []
        rows_s = []
        for t in [sel_home, sel_away]:
            d = stats_match[t]
            rows_f.append({**{'SQUADRA': t}, **d['F']})
            rows_s.append({**{'SQUADRA': t}, **d['S']})
        
        df_f = pd.DataFrame(rows_f).set_index('SQUADRA')
        df_s = pd.DataFrame(rows_s).set_index('SQUADRA')

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        
        sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False, ax=axes[0])
        axes[0].set_title('⚽ DENSITÀ GOL FATTI', fontweight='bold')
        
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=axes[1])
        axes[1].set_title('🛡️ DENSITÀ GOL SUBITI', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
