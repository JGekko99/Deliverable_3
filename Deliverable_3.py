import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numba import njit

# ============================
# FUNZIONI DI CALCOLO E SIMULAZIONE
# ============================

def calcola_capitale_iniziale(costo_sviluppo, costo_infrastruttura, costo_marketing, riserva_operativa):
    """
    Calcola il capitale totale necessario per avviare la divisione.
    """
    return costo_sviluppo + costo_infrastruttura + costo_marketing + riserva_operativa

def proiezioni_finanziarie(ricavo_iniziale, tasso_crescita, rapporto_costi, anni):
    """
    Calcola le proiezioni finanziarie per un numero di anni specificato.
    
    Restituisce un DataFrame con le colonne:
      'Anno', 'Ricavi', 'Costi', 'Utile Netto', 'Utile Cumulativo'
    """
    data = []
    utile_cumulativo = 0
    for anno in range(1, anni + 1):
        ricavo = ricavo_iniziale * ((1 + tasso_crescita) ** (anno - 1))
        costi = ricavo * rapporto_costi
        utile_netto = ricavo - costi
        utile_cumulativo += utile_netto
        data.append((anno, ricavo, costi, utile_netto, utile_cumulativo))
    df = pd.DataFrame(data, columns=['Anno', 'Ricavi', 'Costi', 'Utile Netto', 'Utile Cumulativo'])
    return df

def find_break_even(df, capitale_iniziale):
    """
    Trova l'anno in cui l'utile cumulativo supera o eguaglia il capitale iniziale.
    
    Restituisce l'anno oppure None se il break‑even non viene raggiunto.
    """
    be = df[df['Utile Cumulativo'] >= capitale_iniziale]
    if not be.empty:
        return int(be.iloc[0]['Anno'])
    else:
        return None

# ============================
# PREVISIONE CON LIGHTGBM
# ============================

def train_lightgbm_forecast(anni_storici, ricavo_iniziale, tasso_crescita, rapporto_costi, noise=0.05, anni_forecast=5):
    """
    Simula dati storici dei ricavi (caso "normal") e addestra un modello LightGBM
    per prevedere i ricavi dei prossimi 'anni_forecast' anni.
    
    Parameters:
      - anni_storici: numero di anni storici da simulare
      - ricavo_iniziale, tasso_crescita, rapporto_costi: parametri per la simulazione
      - noise: percentuale di rumore casuale da applicare ai dati storici
      - anni_forecast: numero di anni futuri da prevedere (default 5)
    
    Restituisce:
      - X_future: feature degli anni futuri
      - forecast_df: DataFrame con le previsioni (Ricavi, Costi, Utile Netto, Utile Cumulativo)
      - model: il modello addestrato
    """
    # Genera dati storici
    anni = np.arange(1, anni_storici + 1).reshape(-1, 1)
    ricavi = np.array([ricavo_iniziale * ((1 + tasso_crescita) ** (anno - 1)) for anno in range(1, anni_storici + 1)])
    ricavi = ricavi * (1 + np.random.normal(0, noise, size=ricavi.shape))
    
    # Dividi i dati in train e test
    X_train, X_test, y_train, y_test = train_test_split(anni, ricavi, test_size=0.2, random_state=42)
    
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.ravel())
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE del modello LightGBM sui dati storici: {mse:.2f}")
    
    # Previsione per i prossimi 'anni_forecast' anni
    anni_futuri = np.arange(anni_storici + 1, anni_storici + 1 + anni_forecast).reshape(-1, 1)
    ricavi_pred = model.predict(anni_futuri)
    
    data = []
    utile_cumulativo = 0
    for i, anno in enumerate(anni_futuri.flatten(), start=1):
        ricavo = ricavi_pred[i-1]
        costi = ricavo * rapporto_costi
        utile_netto = ricavo - costi
        utile_cumulativo += utile_netto
        data.append((anno, ricavo, costi, utile_netto, utile_cumulativo))
    forecast_df = pd.DataFrame(data, columns=['Anno', 'Ricavi', 'Costi', 'Utile Netto', 'Utile Cumulativo'])
    
    return anni_futuri, forecast_df, model

# ============================
# GRAFICI INTERATTIVI CON PLOTLY
# ============================

def create_interactive_chart(scenari_data, capitale_iniziale):
    """
    Crea un grafico interattivo che mostra Ricavi, Costi, Utile Netto e Utile Cumulativo
    per ciascun scenario, utilizzando un menu per la selezione.
    """
    fig = go.Figure()
    scenari = list(scenari_data.keys())
    
    # Per ogni scenario aggiungiamo 4 tracce (Ricavi, Costi, Utile Netto, Utile Cumulativo)
    for scenario in scenari:
        df = scenari_data[scenario]
        fig.add_trace(go.Scatter(x=df['Anno'], y=df['Ricavi'], mode='lines+markers',
                                 name=f'{scenario.capitalize()} - Ricavi', visible=(scenario=="normal")))
        fig.add_trace(go.Scatter(x=df['Anno'], y=df['Costi'], mode='lines+markers',
                                 name=f'{scenario.capitalize()} - Costi', visible=(scenario=="normal")))
        fig.add_trace(go.Scatter(x=df['Anno'], y=df['Utile Netto'], mode='lines+markers',
                                 name=f'{scenario.capitalize()} - Utile Netto', visible=(scenario=="normal")))
        fig.add_trace(go.Scatter(x=df['Anno'], y=df['Utile Cumulativo'], mode='lines+markers',
                                 name=f'{scenario.capitalize()} - Utile Cumulativo', visible=(scenario=="normal"),
                                 yaxis="y2"))
    
    # Linea per il capitale iniziale
    fig.add_trace(go.Scatter(x=[1, max(df['Anno'])], y=[capitale_iniziale, capitale_iniziale],
                             mode="lines", name="Capitale Iniziale", line=dict(dash="dot"), visible=True, yaxis="y2"))
    
    # Pulsanti per la selezione dello scenario
    steps = []
    n_tracce_per_scenario = 4  # 4 tracce per scenario
    for i, scenario in enumerate(scenari):
        vis = [False] * (len(scenari)*n_tracce_per_scenario + 1)
        start = i * n_tracce_per_scenario
        for j in range(n_tracce_per_scenario):
            vis[start + j] = True
        vis[-1] = True  # la linea del capitale iniziale sempre visibile
        step = dict(method="update",
                    args=[{"visible": vis},
                          {"title": f"Proiezioni Finanziarie - Scenario {scenario.capitalize()}"}],
                    label=scenario.capitalize())
        steps.append(step)
    
    fig.update_layout(
        updatemenus=[dict(
            active=1,
            buttons=steps,
            x=0.1,
            y=1.15,
            xanchor="left",
            yanchor="top"
        )],
        title="Proiezioni Finanziarie - Scenario Normal",
        xaxis=dict(title="Anno"),
        yaxis=dict(title="Valori (EUR)"),
        yaxis2=dict(title="Utile Cumulativo (EUR)", overlaying="y", side="right"),
        legend=dict(x=0.02, y=0.98)
    )
    fig.show()

def create_ml_vs_simulation_chart(sim_df, ml_df, capitale_iniziale):
    """
    Crea un grafico interattivo che confronta la simulazione classica (caso "normal")
    con la previsione ML ottenuta con LightGBM.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim_df['Anno'], y=sim_df['Ricavi'], mode='lines+markers',
                             name='Simulazione - Ricavi'))
    fig.add_trace(go.Scatter(x=sim_df['Anno'], y=sim_df['Utile Cumulativo'], mode='lines+markers',
                             name='Simulazione - Utile Cumulativo', yaxis="y2"))
    fig.add_trace(go.Scatter(x=ml_df['Anno'], y=ml_df['Ricavi'], mode='lines+markers',
                             name='ML Forecast - Ricavi'))
    fig.add_trace(go.Scatter(x=ml_df['Anno'], y=ml_df['Utile Cumulativo'], mode='lines+markers',
                             name='ML Forecast - Utile Cumulativo', yaxis="y2"))
    fig.add_trace(go.Scatter(x=[min(sim_df['Anno']), max(ml_df['Anno'])],
                             y=[capitale_iniziale, capitale_iniziale],
                             mode="lines", name="Capitale Iniziale", line=dict(dash="dot"), yaxis="y2"))
    
    fig.update_layout(
        title="Confronto: Simulazione vs Previsione ML (Normal Scenario)",
        xaxis=dict(title="Anno"),
        yaxis=dict(title="Ricavi (EUR)"),
        yaxis2=dict(title="Utile Cumulativo (EUR)", overlaying="y", side="right"),
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified"
    )
    fig.show()

# ============================
# SIMULAZIONE MONTE CARLO CON NUMBA
# ============================

@njit
def montecarlo_simulation(ricavo_iniziale, tasso_crescita, rapporto_costi, anni, num_simulations, volatility):
    """
    Esegue la simulazione Monte Carlo per i flussi finanziari.
    Ogni simulazione evolve in maniera stocastica:
      rev[t] = rev[t-1] * (1 + tasso_crescita + shock)
    dove shock ~ N(0, volatility).
    
    Restituisce due array:
      - revenues: matrice (num_simulations x anni) dei ricavi
      - cum_net: matrice (num_simulations x anni) dell'utile netto cumulativo
    """
    revenues = np.zeros((num_simulations, anni))
    cum_net = np.zeros((num_simulations, anni))
    for sim in range(num_simulations):
        rev = ricavo_iniziale
        cum = 0.0
        for t in range(anni):
            if t > 0:
                shock = np.random.normal(0, volatility)
                growth = tasso_crescita + shock
                rev = rev * (1 + growth)
            revenues[sim, t] = rev
            costi = rev * rapporto_costi
            net = rev - costi
            cum += net
            cum_net[sim, t] = cum
    return revenues, cum_net

@njit
def compute_break_even_years(cum_net, capitale_iniziale):
    """
    Per ogni simulazione, determina l'anno in cui l'utile cumulativo raggiunge o supera il capitale iniziale.
    Restituisce un array di anni (se il break‑even non viene raggiunto, viene restituito -1).
    """
    num_sim, anni = cum_net.shape
    break_even_years = -1 * np.ones(num_sim, dtype=np.int32)
    for sim in range(num_sim):
        for t in range(anni):
            if cum_net[sim, t] >= capitale_iniziale:
                break_even_years[sim] = t + 1  # anni a partire da 1
                break
    return break_even_years

def run_montecarlo_simulations(scenari, anni, num_simulations, volatility_params, capitale_iniziale):
    """
    Esegue la simulazione Monte Carlo per ciascun scenario.
    
    Restituisce un dizionario con, per ogni scenario, le matrici:
      - 'revenues': ricavi simulati
      - 'cum_net': utile netto cumulativo simulato
      - 'avg_cum_net': percorso medio (media su tutte le simulazioni)
      - 'break_even_years': array degli anni di break‑even per ogni simulazione
    """
    results = {}
    for scenario, params in scenari.items():
        vol = volatility_params.get(scenario, 0.15)
        revenues, cum_net = montecarlo_simulation(params["ricavo_iniziale"], params["tasso_crescita"], 
                                                   params["rapporto_costi"], anni, num_simulations, vol)
        break_even_years = compute_break_even_years(cum_net, capitale_iniziale)
        avg_cum_net = np.mean(cum_net, axis=0)
        results[scenario] = {
            "revenues": revenues,
            "cum_net": cum_net,
            "avg_cum_net": avg_cum_net,
            "break_even_years": break_even_years
        }
    return results

def create_mc_interactive_chart(mc_results, capitale_iniziale, anni):
    """
    Crea un grafico interattivo che mostra il percorso medio dell'utile cumulativo 
    e l'intervallo (percentili 5°-95°) per ciascun scenario, con la linea del capitale iniziale.
    """
    fig = go.Figure()
    scenarios = list(mc_results.keys())
    
    for scenario in scenarios:
        cum_net = mc_results[scenario]["cum_net"]
        avg = np.mean(cum_net, axis=0)
        lower = np.percentile(cum_net, 5, axis=0)
        upper = np.percentile(cum_net, 95, axis=0)
        anni_array = np.arange(1, anni+1)
        fig.add_trace(go.Scatter(
            x=anni_array, y=avg,
            mode='lines+markers',
            name=f'{scenario.capitalize()} - Media Utile Cumulativo'
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([anni_array, anni_array[::-1]]),
            y=np.concatenate([lower, upper[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name=f'{scenario.capitalize()} - Intervallo 5°-95°'
        ))
    fig.add_trace(go.Scatter(
        x=[1, anni],
        y=[capitale_iniziale, capitale_iniziale],
        mode="lines",
        name="Capitale Iniziale",
        line=dict(dash="dot", color="red")
    ))
    
    fig.update_layout(
        title="Monte Carlo Simulation - Utile Cumulativo Medio e Intervallo (5°-95°)",
        xaxis=dict(title="Anno"),
        yaxis=dict(title="Utile Cumulativo (EUR)"),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    fig.show()

# ============================
# FUNZIONE MAIN: INTEGRA TUTTO
# ============================

def main():
    # --- 1. Fabbisogno Finanziario ---
    # Stima dei costi per avviare la divisione (in EUR)
    costo_sviluppo = 200000        # Sviluppo software/AI per il chatbot
    costo_infrastruttura = 150000   # Hardware e infrastruttura IT
    costo_marketing = 100000        # Campagne di marketing e branding
    riserva_operativa = 50000       # Riserva operativa iniziale
    
    capitale_iniziale = calcola_capitale_iniziale(costo_sviluppo, costo_infrastruttura, costo_marketing, riserva_operativa)
    print("ANALISI DEL FABBISOGNO FINANZIARIO")
    print(f"Capitale necessario per avviare la divisione: {capitale_iniziale} EUR")
    
    # --- 2. Proiezioni Finanziarie a 5 anni ---
    anni_simulazione = 5
    scenari = {
        "worst": {
            "ricavo_iniziale": 80000,
            "tasso_crescita": 0.10,
            "rapporto_costi": 0.9
        },
        "normal": {
            "ricavo_iniziale": 100000,
            "tasso_crescita": 0.25,
            "rapporto_costi": 0.8
        },
        "best": {
            "ricavo_iniziale": 120000,
            "tasso_crescita": 0.40,
            "rapporto_costi": 0.7
        }
    }
    
    # Simulazione deterministica per ciascun scenario (5 anni)
    scenari_data = {}
    for scenario, params in scenari.items():
        df = proiezioni_finanziarie(params["ricavo_iniziale"], params["tasso_crescita"], params["rapporto_costi"], anni_simulazione)
        scenari_data[scenario] = df
        print(f"\nScenario {scenario.capitalize()} - Proiezioni a 5 anni:")
        print(df.to_string(index=False))
        be_year = find_break_even(df, capitale_iniziale)
        if be_year:
            print(f"-> Break-Even raggiunto all'anno: {be_year}")
        else:
            print("-> Break-Even non raggiunto nei dati simulati.")
    
    # --- 3. Previsione ML con LightGBM per il caso "normal" ---
    anni_storici = 5
    normal_params = scenari["normal"]
    normal_df = proiezioni_finanziarie(normal_params["ricavo_iniziale"], normal_params["tasso_crescita"], normal_params["rapporto_costi"], anni_storici)
    print("\nDati storici simulati per il caso Normal:")
    print(normal_df.to_string(index=False))
    
    # Prevediamo i prossimi 5 anni (forecast)
    X_future, ml_forecast_df, model = train_lightgbm_forecast(anni_storici, normal_params["ricavo_iniziale"],
                                                              normal_params["tasso_crescita"], normal_params["rapporto_costi"],
                                                              noise=0.05, anni_forecast=5)
    print("\nPrevisioni ML (LightGBM) per il caso Normal (5 anni):")
    print(ml_forecast_df.to_string(index=False))
    
    # --- 4. Visualizzazione interattiva delle simulazioni deterministiche e ML ---
    create_interactive_chart(scenari_data, capitale_iniziale)
    create_ml_vs_simulation_chart(scenari_data["normal"], ml_forecast_df, capitale_iniziale)
    
    # --- 5. Simulazione Monte Carlo con Numba per 5 anni ---
    num_simulations = 1000  # Numero di simulazioni
    # Volatilità per ciascun scenario
    volatility_params = {
        "worst": 0.20,
        "normal": 0.15,
        "best": 0.10
    }
    
    mc_results = run_montecarlo_simulations(scenari, anni_simulazione, num_simulations, volatility_params, capitale_iniziale)
    create_mc_interactive_chart(mc_results, capitale_iniziale, anni_simulazione)
    
    # Statistiche sul break‑even nelle simulazioni Monte Carlo
    for scenario in mc_results:
        break_even_years = mc_results[scenario]["break_even_years"]
        reached = break_even_years[break_even_years != -1]
        if reached.size > 0:
            perc = (reached.size / num_simulations) * 100
            avg_year = np.mean(reached)
            print(f"\nScenario {scenario.capitalize()} - {perc:.1f}% delle simulazioni raggiungono il break‑even, "
                  f"con un anno medio di break‑even pari a {avg_year:.1f}.")
        else:
            print(f"\nScenario {scenario.capitalize()} - Nessuna simulazione ha raggiunto il break‑even.")

if __name__ == "__main__":
    main()
