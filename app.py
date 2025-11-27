import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Analisi Calcio Pro", layout="wide")

st.title("⚽ Dashboard Analisi Calcio Avanzata")
st.markdown("Carica il file dei dati e analizza le partite.")

# ==========================================
# 1. CARICAMENTO FILE
# ==========================================
uploaded_file = st.file_uploader("Carica il file CSV o Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Tenta lettura CSV con separatore automatico
        try:
            # Legge prima riga per capire separatore
            line = uploaded_file.readline().decode('latin1')
            uploaded_file.seek(0) # Torna all'inizio
            sep = ';' if line.count(';') > line.count(',') else ','
            
            df = pd.read_csv(uploaded_file, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
        except:
            # Prova come Excel
            df = pd.read_excel(uploaded_file)

        # Pulizia e Rinomina Colonne
        df.columns = df.columns.astype(str).str.strip().str.upper()
        df = df.loc[:, ~df.columns.duplicated()]

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

        st.success(f"File caricato con successo! {len(df)} righe trovate.")
        
        # ==========================================
        # 2. INTERFACCIA SELEZIONE
        # ==========================================
        col1, col2, col3 = st.columns(3)
        
        with col1:
            leghe = sorted(df['ID_LEGA'].unique())
            sel_lega = st.selectbox("Seleziona Campionato", leghe)
        
        df_league = df[df['ID_LEGA'] == sel_lega].copy()
        teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())
        
        with col2:
            sel_home = st.selectbox("Squadra Casa", teams, index=0)
        
        with col3:
            # Cerca di selezionare una squadra diversa dalla home per comodità
            idx_away = 1 if len(teams) > 1 else 0
            sel_away = st.selectbox("Squadra Ospite", teams, index=idx_away)

        if st.button("🚀 AVVIA ANALISI"):
            st.divider()
            st.subheader(f"⚔️ Analisi: {sel_home} vs {sel_away}")
            
            # --- ELABORAZIONE ---
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

            goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
            goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
            match_h, match_a = 0, 0
            times_h, times_a = [], []
            teams_heatmap = {}

            for _, row in df_league.iterrows():
                h, a = row['CASA'], row['OSPITE']
                min_h = get_minutes(row.get(c_h))
                min_a = get_minutes(row.get(c_a))
                
                for t in [h, a]:
                    if t not in teams_heatmap:
                        teams_heatmap[t] = {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
                
                # Heatmap data
                for m in min_h:
                    idx = min(5, (m-1)//15)
                    if m > 45 and m <= 60 and idx < 3: idx = 3
                    interval = intervals[idx]
                    teams_heatmap[h]['F'][interval] += 1
                    teams_heatmap[a]['S'][interval] += 1
                
                for m in min_a:
                    idx = min(5, (m-1)//15)
                    if m > 45 and m <= 60 and idx < 3: idx = 3
                    interval = intervals[idx]
                    teams_heatmap[a]['F'][interval] += 1
                    teams_heatmap[h]['S'][interval] += 1

                # Specific Match Data
                if h == sel_home:
                    match_h += 1
                    goals_h['FT'] += len(min_h)
                    goals_h['HT'] += len([x for x in min_h if x <= 45])
                    goals_h['S_FT'] += len(min_a)
                    goals_h['S_HT'] += len([x for x in min_a if x <= 45])
                    times_h.extend(min_h)
                
                if a == sel_away:
                    match_a += 1
                    goals_a['FT'] += len(min_a)
                    goals_a['HT'] += len([x for x in min_a if x <= 45])
                    goals_a['S_FT'] += len(min_h)
                    goals_a['S_HT'] += len([x for x in min_h if x <= 45])
                    times_a.extend(min_a)
            
            # Medie
            avg_h_ft = goals_h['FT'] / match_h if match_h else 0
            avg_h_ht = goals_h['HT'] / match_h if match_h else 0
            avg_h_conc_ft = goals_h['S_FT'] / match_h if match_h else 0
            avg_h_conc_ht = goals_h['S_HT'] / match_h if match_h else 0

            avg_a_ft = goals_a['FT'] / match_a if match_a else 0
            avg_a_ht = goals_a['HT'] / match_a if match_a else 0
            avg_a_conc_ft = goals_a['S_FT'] / match_a if match_a else 0
            avg_a_conc_ht = goals_a['S_HT'] / match_a if match_a else 0

            # Tabella Medie
            st.write("### 📊 Medie Gol (Casa vs Fuori)")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric(label=f"{sel_home} (Casa)", value=f"{avg_h_ft:.2f} GF", delta=f"-{avg_h_conc_ft:.2f} GS")
            col_res2.metric(label=f"{sel_away} (Ospite)", value=f"{avg_a_ft:.2f} GF", delta=f"-{avg_a_conc_ft:.2f} GS")
            
            st.info(f"**Primo Tempo:** {sel_home} segna {avg_h_ht:.2f} | {sel_away} segna {avg_a_ht:.2f}")

            # --- GRAFICI ---
            tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol (Kaplan-Meier)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])

            with tab1:
                if times_h and times_a:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    kmf_h = KaplanMeierFitter()
                    kmf_a = KaplanMeierFitter()
                    kmf_h.fit(times_h, label=f'{sel_home} Gol')
                    kmf_a.fit(times_a, label=f'{sel_away} Gol')
                    kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
                    kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
                    plt.title('Probabilità di 0-0 nel tempo')
                    plt.grid(True, alpha=0.3)
                    plt.axvline(45, color='green', linestyle='--')
                    st.pyplot(fig)
                else:
                    st.warning("Dati insufficienti per il grafico del ritmo.")

            # Preparazione Heatmap Dataframes
            rows_f = []
            rows_s = []
            for t in [sel_home, sel_away]:
                if t in teams_heatmap:
                    d = teams_heatmap[t]
                    rows_f.append({**{'SQUADRA': t}, **d['F']})
                    rows_s.append({**{'SQUADRA': t}, **d['S']})
            
            if rows_f:
                df_f = pd.DataFrame(rows_f).set_index('SQUADRA')
                df_s = pd.DataFrame(rows_s).set_index('SQUADRA')
                
                with tab2:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False, ax=ax)
                    plt.title("Densità Gol Fatti")
                    st.pyplot(fig)
                
                with tab3:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=ax)
                    plt.title("Densità Gol Subiti")
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Errore durante l'elaborazione: {e}")
