import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import lightgbm as lgb
from numba import jit
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
import io
import base64
from scipy.stats import norm

# Impostazioni di stile
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

# Funzione per creare grafici HTML interattivi
def get_interactive_plot():
    # Titolo e introduzione
    html_content = """
    <div style="width:100%; background-color:#f8f9fa; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:#2c3e50; text-align:center;">Analisi del Fabbisogno Finanziario</h1>
        <h2 style="color:#34495e; text-align:center;">Fintech con Chatbot AI per clienti under 35</h2>
        <p style="color:#7f8c8d; text-align:center; font-style:italic;">
            Analisi deterministica, previsioni ML e simulazioni Monte Carlo per 5 anni
        </p>
    </div>
    """
    return html_content

# 1. PARAMETRI DI BASE PER LA SIMULAZIONE
# Definiamo i parametri basati sul contesto della fintech di chatbot AI bancario
# Investimento iniziale e parametri operativi
params = {
    # Costi di avvio
    'sviluppo_piattaforma': 1200000,  # Sviluppo piattaforma AI e chatbot
    'infrastruttura_it': 800000,      # Server, cloud, sicurezza
    'licenze_software': 300000,       # Licenze software e API
    'conformita_regolamentare': 500000,  # Conformità GDPR, normative bancarie
    'marketing_iniziale': 600000,     # Lancio e acquisizione clienti iniziale
    
    # Costi operativi annuali (base)
    'personale_tecnico': 1500000,     # Team tecnico e data scientists
    'personale_supporto': 800000,     # Supporto clienti e operazioni
    'personale_conformita': 600000,   # Team legale e conformità
    'cloud_hosting': 400000,          # Costi cloud e hosting
    'manutenzione_sistema': 300000,   # Manutenzione e aggiornamenti
    'marketing_annuale': 800000,      # Marketing e acquisizione clienti
    'formazione': 200000,             # Formazione personale
    
    # Parametri ricavi
    'cliente_base_annuo': 50,         # Ricavo medio per cliente all'anno (servizi base)
    'cliente_premium_annuo': 120,     # Ricavo medio per cliente premium
    'transazioni_percentuale': 0.5,   # % sulle transazioni
    'volume_medio_transazione': 1000, # Volume medio transazione per cliente
    'frequenza_transazioni': 24,      # Numero medio transazioni annue per cliente
    
    # Crescita clienti (clienti totali alla fine di ogni anno)
    'clienti_y1': 25000,
    'clienti_y2': 75000,
    'clienti_y3': 150000,
    'clienti_y4': 250000,
    'clienti_y5': 400000,
    
    # % clienti premium
    'premium_ratio_y1': 0.08,
    'premium_ratio_y2': 0.12, 
    'premium_ratio_y3': 0.15,
    'premium_ratio_y4': 0.18,
    'premium_ratio_y5': 0.22,
    
    # Tassi di crescita/riduzione costi
    'costi_fissi_crescita': 0.03,     # Crescita annua costi fissi
    'efficienza_scale': 0.04,         # Efficienza di scala con crescita clienti
    
    # Variazioni per scenari
    'worst_factor': 0.7,              # Fattore moltiplicativo per scenario pessimistico
    'best_factor': 1.3,               # Fattore moltiplicativo per scenario ottimistico
    
    # Simulazione Monte Carlo
    'num_simulations': 1000,          # Numero di simulazioni Monte Carlo
    'volatility': 0.25,               # Volatilità per simulazione Monte Carlo
}

# 2. CALCOLO DELL'INVESTIMENTO INIZIALE
initial_investment = (params['sviluppo_piattaforma'] + 
                      params['infrastruttura_it'] + 
                      params['licenze_software'] + 
                      params['conformita_regolamentare'] + 
                      params['marketing_iniziale'])

# 3. SIMULAZIONE DETERMINISTICA DEI FLUSSI FINANZIARI
def simulate_financials(scenario='normal'):
    # Definizione moltiplicatori in base allo scenario
    if scenario == 'worst':
        factor = params['worst_factor']
    elif scenario == 'best':
        factor = params['best_factor']
    else:  # normal
        factor = 1.0
    
    # Definizione struttura dati risultati
    years = 5
    results = {
        'year': list(range(1, years+1)),
        'customers': [],
        'premium_customers': [],
        'regular_customers': [],
        'revenue_base': [],
        'revenue_premium': [],
        'revenue_transactions': [],
        'revenue_total': [],
        'cost_personnel': [],
        'cost_tech': [],
        'cost_marketing': [],
        'cost_other': [],
        'cost_total': [],
        'ebitda': [],
        'ebitda_margin': [],
        'cumulative_cashflow': []
    }
    
    # Calcolo per ogni anno
    cumulative_cf = -initial_investment
    
    for year in range(1, years+1):
        # Clienti
        customers_key = f'clienti_y{year}'
        premium_ratio_key = f'premium_ratio_y{year}'
        
        # Applica il fattore di scenario ai clienti
        customers = params[customers_key] * factor
        premium_ratio = params[premium_ratio_key]
        premium_customers = customers * premium_ratio
        regular_customers = customers - premium_customers
        
        # Calcolo ricavi
        revenue_base = regular_customers * params['cliente_base_annuo']
        revenue_premium = premium_customers * params['cliente_premium_annuo']
        
        # Ricavi da transazioni (crescono con l'uso del servizio)
        transaction_volume = customers * params['volume_medio_transazione'] * params['frequenza_transazioni']
        transaction_revenue = transaction_volume * params['transazioni_percentuale'] / 100
        
        # Aggiustamento ricavi per scenario
        revenue_base *= factor
        revenue_premium *= factor
        transaction_revenue *= factor
        
        total_revenue = revenue_base + revenue_premium + transaction_revenue
        
        # Calcolo costi (con efficienze di scala)
        efficiency_scale = 1.0 - (params['efficienza_scale'] * (year-1)) if year > 1 else 1.0
        growth_factor = (1 + params['costi_fissi_crescita']) ** (year-1)
        
        # Costi del personale
        cost_personnel = (params['personale_tecnico'] + 
                          params['personale_supporto'] + 
                          params['personale_conformita']) * growth_factor * efficiency_scale
        
        # Costi tecnologici
        cost_tech = (params['cloud_hosting'] + 
                     params['manutenzione_sistema']) * growth_factor * efficiency_scale
        
        # Costi marketing (crescono con l'acquisizione clienti)
        marketing_scale_factor = 1.0 + (0.1 * (year-1))
        cost_marketing = params['marketing_annuale'] * marketing_scale_factor
        
        # Altri costi
        cost_other = params['formazione'] * growth_factor
        
        # Totale costi
        total_cost = cost_personnel + cost_tech + cost_marketing + cost_other
        
        # EBITDA
        ebitda = total_revenue - total_cost
        ebitda_margin = (ebitda / total_revenue * 100) if total_revenue > 0 else 0
        
        # Flusso di cassa cumulativo
        cumulative_cf += ebitda
        
        # Salvataggio risultati
        results['customers'].append(customers)
        results['premium_customers'].append(premium_customers)
        results['regular_customers'].append(regular_customers)
        results['revenue_base'].append(revenue_base)
        results['revenue_premium'].append(revenue_premium)
        results['revenue_transactions'].append(transaction_revenue)
        results['revenue_total'].append(total_revenue)
        results['cost_personnel'].append(cost_personnel)
        results['cost_tech'].append(cost_tech)
        results['cost_marketing'].append(cost_marketing)
        results['cost_other'].append(cost_other)
        results['cost_total'].append(total_cost)
        results['ebitda'].append(ebitda)
        results['ebitda_margin'].append(ebitda_margin)
        results['cumulative_cashflow'].append(cumulative_cf)
    
    return pd.DataFrame(results)

# Simulazione per i tre scenari
df_normal = simulate_financials('normal')
df_worst = simulate_financials('worst')
df_best = simulate_financials('best')

# 4. CREAZIONE DATI STORICI SINTETICI PER LIGHTGBM
# Creiamo dati storici sintetici per il modello di ML
def create_synthetic_data():
    # Creiamo dati storici mensili per 36 mesi (3 anni) precedenti
    np.random.seed(42)
    months = 36
    dates = [dt.datetime.now() - dt.timedelta(days=30*i) for i in range(months, 0, -1)]
    
    # Base di clienti iniziale e crescita
    initial_customers = 5000
    customers = [initial_customers]
    
    for i in range(1, months):
        # Crescita con stagionalità e trend
        growth_rate = 0.05 + 0.02 * np.sin(i/12 * 2 * np.pi)  # Stagionalità annuale
        noise = np.random.normal(0, 0.01)
        new_customers = customers[-1] * (1 + growth_rate + noise)
        customers.append(new_customers)
    
    # Creazione features
    df = pd.DataFrame({
        'date': dates,
        'customers': customers,
        'month': [d.month for d in dates],
        'year': [d.year for d in dates],
        'day_of_year': [d.timetuple().tm_yday for d in dates],
        # Correzione: calcola il trimestre dal mese
        'quarter': [(d.month - 1) // 3 + 1 for d in dates]
    })
    
    # Aggiungiamo più features
    df['trend'] = np.arange(len(df))
    df['season_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
    df['season_cos'] = np.cos(df['month'] * (2 * np.pi / 12))
    
    # Ricavi per cliente (simula l'aumento del valore nel tempo)
    base_revenue_per_customer = 4  # €/mese
    df['revenue_per_customer'] = base_revenue_per_customer * (1 + 0.005 * df['trend'])
    
    # Calcolo ricavi totali mensili
    df['revenue'] = df['customers'] * df['revenue_per_customer']
    
    return df

# 5. TRAINING DEL MODELLO LIGHTGBM PER PREVISIONE RICAVI
def train_and_predict_lightgbm():
    # Creazione dati storici sintetici
    hist_data = create_synthetic_data()
    
    # Feature da utilizzare
    features = ['month', 'quarter', 'trend', 'season_sin', 'season_cos', 'customers']
    target = 'revenue'
    
    # Preparazione dataset
    X_train = hist_data[features]
    y_train = hist_data[target]
    
    # Addestramento modello LightGBM
    params_lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params_lgb)
    model.fit(X_train, y_train)
    
    # Creazione dati futuri per la previsione (60 mesi, 5 anni)
    future_months = 60
    last_date = hist_data['date'].iloc[-1]
    future_dates = [last_date + dt.timedelta(days=30*i) for i in range(1, future_months+1)]
    
    # Crescita di clienti nel futuro (basata sui tre scenari)
    customers_normal = []
    customers_worst = []
    customers_best = []
    
    last_customers = hist_data['customers'].iloc[-1]
    for i in range(future_months):
        year = i // 12
        if year == 0:
            target_year_end = params['clienti_y1'] / 12
            target_year_end_worst = params['clienti_y1'] * params['worst_factor'] / 12
            target_year_end_best = params['clienti_y1'] * params['best_factor'] / 12
        else:
            curr_key = f'clienti_y{min(year+1, 5)}'
            prev_key = f'clienti_y{min(year, 5)}'
            monthly_growth = (params[curr_key] - params[prev_key]) / 12
            target_year_end = monthly_growth
            target_year_end_worst = monthly_growth * params['worst_factor']
            target_year_end_best = monthly_growth * params['best_factor']
        
        if i == 0:
            customers_normal.append(last_customers + target_year_end)
            customers_worst.append(last_customers + target_year_end_worst)
            customers_best.append(last_customers + target_year_end_best)
        else:
            customers_normal.append(customers_normal[-1] + target_year_end)
            customers_worst.append(customers_worst[-1] + target_year_end_worst)
            customers_best.append(customers_best[-1] + target_year_end_best)
    
    # Dataframe futuro per i tre scenari
    future_data_normal = pd.DataFrame({
        'date': future_dates,
        'month': [d.month for d in future_dates],
        'year': [d.year for d in future_dates],
        'quarter': [(d.month - 1) // 3 + 1 for d in future_dates],  # Correzione qui
        'trend': np.arange(len(hist_data), len(hist_data) + future_months),
        'customers': customers_normal
})
    
    future_data_worst = future_data_normal.copy()
    future_data_worst['customers'] = customers_worst
    
    future_data_best = future_data_normal.copy()
    future_data_best['customers'] = customers_best
    
    # Aggiunta features stagionali
    for df in [future_data_normal, future_data_worst, future_data_best]:
        df['season_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
        df['season_cos'] = np.cos(df['month'] * (2 * np.pi / 12))
    
    # Previsione per i tre scenari
    X_future_normal = future_data_normal[features]
    X_future_worst = future_data_worst[features]
    X_future_best = future_data_best[features]
    
    future_data_normal['revenue_predicted'] = model.predict(X_future_normal)
    future_data_worst['revenue_predicted'] = model.predict(X_future_worst)
    future_data_best['revenue_predicted'] = model.predict(X_future_best)
    
    # Aggregazione per anno
    future_data_normal['year_num'] = future_data_normal['date'].dt.year - future_data_normal['date'].dt.year.min() + 1
    future_data_worst['year_num'] = future_data_worst['date'].dt.year - future_data_worst['date'].dt.year.min() + 1
    future_data_best['year_num'] = future_data_best['date'].dt.year - future_data_best['date'].dt.year.min() + 1
    
    annual_normal = future_data_normal.groupby('year_num')['revenue_predicted'].sum().reset_index()
    annual_worst = future_data_worst.groupby('year_num')['revenue_predicted'].sum().reset_index()
    annual_best = future_data_best.groupby('year_num')['revenue_predicted'].sum().reset_index()
    
    return hist_data, future_data_normal, future_data_worst, future_data_best, annual_normal, annual_worst, annual_best

# 6. SIMULAZIONE MONTE CARLO CON NUMBA
@jit(nopython=True)
def monte_carlo_simulation(starting_capital, years, num_sims, growth_rates, volatilities):
    """
    Simulazione Monte Carlo accelerata con Numba
    """
    paths = np.zeros((years, num_sims))
    
    for sim in range(num_sims):
        capital = starting_capital
        for year in range(years):
            growth = growth_rates[year]
            vol = volatilities[year]
            
            # Simulazione con distribuzione log-normale
            z = np.random.normal(0, 1)
            return_rate = np.exp((growth - 0.5 * vol**2) + vol * z) - 1
            
            capital += capital * return_rate
            paths[year, sim] = capital
    
    return paths

# Esecuzione simulazione Monte Carlo per i tre scenari
def run_monte_carlo():
    years = 5
    num_sims = params['num_simulations']
    
    # Ricaviamo tassi di crescita annuali dai dati deterministici
    normal_growth_rates = []
    for i in range(len(df_normal)-1):
        if df_normal['revenue_total'][i] > 0:
            growth = (df_normal['revenue_total'][i+1] / df_normal['revenue_total'][i]) - 1
        else:
            growth = 0.5  # valore di default per il primo anno
        normal_growth_rates.append(growth)
    normal_growth_rates.append(normal_growth_rates[-1])  # replica ultimo anno
    
    # Adattamento per gli scenari worst e best
    worst_growth_rates = [max(0.6 * rate, -0.1) for rate in normal_growth_rates]  # limita il minimo a -10%
    best_growth_rates = [1.4 * rate for rate in normal_growth_rates]
    
    # Volatilità decrescente con gli anni (rappresenta la riduzione dell'incertezza)
    base_vol = params['volatility']
    volatilities = [base_vol * (1 - 0.1*i) for i in range(years)]
    
    # Capitale iniziale (primo anno di ricavi dal modello deterministico)
    starting_capital_normal = df_normal['revenue_total'][0]
    starting_capital_worst = df_worst['revenue_total'][0]
    starting_capital_best = df_best['revenue_total'][0]
    
    # Esecuzione simulazioni per i tre scenari
    np.random.seed(42)  # Per riproducibilità
    mc_paths_normal = monte_carlo_simulation(
        starting_capital_normal, years, num_sims, 
        np.array(normal_growth_rates), np.array(volatilities)
    )
    
    np.random.seed(42)
    mc_paths_worst = monte_carlo_simulation(
        starting_capital_worst, years, num_sims, 
        np.array(worst_growth_rates), np.array(volatilities)
    )
    
    np.random.seed(42)
    mc_paths_best = monte_carlo_simulation(
        starting_capital_best, years, num_sims, 
        np.array(best_growth_rates), np.array(volatilities)
    )
    
    # Calcolo statistiche
    mc_results = {
        'normal': {
            'mean': np.mean(mc_paths_normal, axis=1),
            'median': np.median(mc_paths_normal, axis=1),
            'p10': np.percentile(mc_paths_normal, 10, axis=1),
            'p90': np.percentile(mc_paths_normal, 90, axis=1),
            'paths': mc_paths_normal
        },
        'worst': {
            'mean': np.mean(mc_paths_worst, axis=1),
            'median': np.median(mc_paths_worst, axis=1),
            'p10': np.percentile(mc_paths_worst, 10, axis=1),
            'p90': np.percentile(mc_paths_worst, 90, axis=1),
            'paths': mc_paths_worst
        },
        'best': {
            'mean': np.mean(mc_paths_best, axis=1),
            'median': np.median(mc_paths_best, axis=1),
            'p10': np.percentile(mc_paths_best, 10, axis=1),
            'p90': np.percentile(mc_paths_best, 90, axis=1),
            'paths': mc_paths_best
        }
    }
    
    return mc_results

# 7. CALCOLO DEL BREAK-EVEN POINT PER OGNI SCENARIO
def calculate_breakeven():
    # Calcolo del punto di break-even per ogni scenario
    be_normal = None
    be_worst = None
    be_best = None
    
    # Per lo scenario normale
    for i, cf in enumerate(df_normal['cumulative_cashflow']):
        if cf >= 0 and i > 0:
            # Interpolazione lineare per stimare il momento preciso di break-even
            prev_cf = df_normal['cumulative_cashflow'][i-1]
            if prev_cf < 0:
                y1, y2 = prev_cf, cf
                x1, x2 = i, i+1
                be_normal = x1 + (0 - y1) / (y2 - y1)
                break
    
    # Per lo scenario worst
    for i, cf in enumerate(df_worst['cumulative_cashflow']):
        if cf >= 0 and i > 0:
            prev_cf = df_worst['cumulative_cashflow'][i-1]
            if prev_cf < 0:
                y1, y2 = prev_cf, cf
                x1, x2 = i, i+1
                be_worst = x1 + (0 - y1) / (y2 - y1)
                break
    
    # Per lo scenario best
    for i, cf in enumerate(df_best['cumulative_cashflow']):
        if cf >= 0 and i > 0:
            prev_cf = df_best['cumulative_cashflow'][i-1]
            if prev_cf < 0:
                y1, y2 = prev_cf, cf
                x1, x2 = i, i+1
                be_best = x1 + (0 - y1) / (y2 - y1)
                break
    
    return {'normal': be_normal, 'worst': be_worst, 'best': be_best}

# 8. CREAZIONE GRAFICI INTERATTIVI CON PLOTLY
def create_plotly_figures():
    # Training e previsione con LightGBM
    hist_data, future_normal, future_worst, future_best, annual_normal, annual_worst, annual_best = train_and_predict_lightgbm()
    
    # Simulazione Monte Carlo
    mc_results = run_monte_carlo()
    
    # Calcolo break-even
    breakeven = calculate_breakeven()
    
    # 1. Grafico dell'investimento iniziale
    fig_investment = go.Figure()
    
    # Preparazione dati
    investment_categories = ['Sviluppo Piattaforma', 'Infrastruttura IT', 'Licenze Software', 
                              'Conformità Regolamentare', 'Marketing Iniziale']
    investment_values = [params['sviluppo_piattaforma'], params['infrastruttura_it'], 
                          params['licenze_software'], params['conformita_regolamentare'],
                          params['marketing_iniziale']]
    
    fig_investment.add_trace(go.Bar(
        x=investment_categories,
        y=investment_values,
        text=[f'€{x:,.0f}' for x in investment_values],
        textposition='auto',
        marker_color=['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 
                      'rgba(44, 160, 44, 0.8)', 'rgba(214, 39, 40, 0.8)', 
                      'rgba(148, 103, 189, 0.8)']
    ))
    
    fig_investment.update_layout(
        title='Investimento Iniziale: €{:,.0f}'.format(initial_investment),
        xaxis_title='Categoria',
        yaxis_title='Importo (€)',
        template='plotly_white',
        height=500
    )
    
    # 2. Grafico dell'andamento clienti
    fig_customers = go.Figure()
    
    fig_customers.add_trace(go.Bar(
        x=df_normal['year'],
        y=df_normal['regular_customers'],
        name='Clienti Base',
        marker_color='rgba(55, 83, 109, 0.7)'
    ))
    
    fig_customers.add_trace(go.Bar(
        x=df_normal['year'],
        y=df_normal['premium_customers'],
        name='Clienti Premium',
        marker_color='rgba(26, 118, 255, 0.7)'
    ))
    
    # Linee per gli scenari worst e best
    fig_customers.add_trace(go.Scatter(
        x=df_worst['year'],
        y=df_worst['customers'],
        name='Scenario Pessimistico',
        mode='lines+markers',
        line=dict(color='rgba(219, 64, 82, 0.7)', dash='dot'),
        marker=dict(size=8)
    ))
    
    fig_customers.add_trace(go.Scatter(
        x=df_best['year'],
        y=df_best['customers'],
        name='Scenario Ottimistico',
        mode='lines+markers',
        line=dict(color='rgba(0, 128, 0, 0.7)', dash='dot'),
        marker=dict(size=8)
    ))
    
    fig_customers.update_layout(
        barmode='stack',
        title='Evoluzione Base Clienti nei 3 Scenari',
        xaxis_title='Anno',
        yaxis_title='Numero Clienti',
        legend_title='Tipologia',
        template='plotly_white',
        height=500
    )
    
    # 3. Grafico dei ricavi vs costi
    fig_revenue_cost = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Ricavi
    fig_revenue_cost.add_trace(
        go.Bar(
            x=df_normal['year'],
            y=df_normal['revenue_total'],
            name='Ricavi Totali',
            marker_color='rgba(0, 128, 0, 0.7)'
        ),
        secondary_y=False
    )
    
    # Costi
    fig_revenue_cost.add_trace(
        go.Bar(
            x=df_normal['year'],
            y=df_normal['cost_total'],
            name='Costi Totali',
            marker_color='rgba(219, 64, 82, 0.7)'
        ),
        secondary_y=False
    )
    
    # EBITDA Margin
    fig_revenue_cost.add_trace(
        go.Scatter(
            x=df_normal['year'],
            y=df_normal['ebitda_margin'],
            name='Margine EBITDA (%)',
            mode='lines+markers',
            line=dict(color='rgba(26, 118, 255, 1.0)', width=3),
            marker=dict(size=10)
        ),
        secondary_y=True
    )
    
    fig_revenue_cost.update_layout(
        title='Ricavi vs Costi e Margine EBITDA',
        xaxis_title='Anno',
        legend_title='Metrica',
        template='plotly_white',
        height=500
    )
    
    fig_revenue_cost.update_yaxes(title_text="Importo (€)", secondary_y=False)
    fig_revenue_cost.update_yaxes(title_text="Margine EBITDA (%)", secondary_y=True)
    
    # 4. Grafico del flusso di cassa cumulativo e break-even
    fig_cashflow = go.Figure()
    
    # Linee per i tre scenari
    fig_cashflow.add_trace(go.Scatter(
        x=list(range(6)),  # 0 + 5 anni
        y=[-initial_investment] + df_normal['cumulative_cashflow'].tolist(),
        name='Scenario Normale',
        mode='lines+markers',
        line=dict(color='rgba(26, 118, 255, 1.0)', width=3),
        marker=dict(size=10)
    ))
    
    fig_cashflow.add_trace(go.Scatter(
        x=list(range(6)),
        y=[-initial_investment] + df_worst['cumulative_cashflow'].tolist(),
        name='Scenario Pessimistico',
        mode='lines+markers',
        line=dict(color='rgba(219, 64, 82, 0.7)', dash='dot'),
        marker=dict(size=8)
    ))
    
    fig_cashflow.add_trace(go.Scatter(
        x=list(range(6)),
        y=[-initial_investment] + df_best['cumulative_cashflow'].tolist(),
        name='Scenario Ottimistico',
        mode='lines+markers',
        line=dict(color='rgba(0, 128, 0, 0.7)', dash='dot'),
        marker=dict(size=8)
    ))
    
    # Linea di break-even
    fig_cashflow.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=5,
        y1=0,
        line=dict(
            color="black",
            width=2,
            dash="dashdot",
        ),
    )
    
    # Annotazioni per break-even
    if breakeven['normal'] is not None:
        fig_cashflow.add_annotation(
            x=breakeven['normal'],
            y=50000,
            text=f"Break-even a {breakeven['normal']:.1f} anni",
            showarrow=True,
            arrowhead=3,
            ax=0,
            ay=-40
        )
    
    if breakeven['worst'] is not None:
        fig_cashflow.add_annotation(
            x=breakeven['worst'],
            y=50000,
            text=f"Break-even (worst) a {breakeven['worst']:.1f} anni",
            showarrow=True,
            arrowhead=3,
            ax=0,
            ay=-80
        )
    
    if breakeven['best'] is not None:
        fig_cashflow.add_annotation(
            x=breakeven['best'],
            y=50000,
            text=f"Break-even (best) a {breakeven['best']:.1f} anni",
            showarrow=True,
            arrowhead=3,
            ax=0,
            ay=-120
        )
    
    fig_cashflow.update_layout(
        title='Flusso di Cassa Cumulativo e Break-Even Point',
        xaxis_title='Anno',
        yaxis_title='Flusso di Cassa Cumulativo (€)',
        template='plotly_white',
        height=500
    )
    
    # 5. Grafico della previsione ML (revenue mensile)
    fig_ml = go.Figure()
    
    # Dati storici
    fig_ml.add_trace(go.Scatter(
        x=hist_data['date'],
        y=hist_data['revenue'],
        name='Dati Storici',
        mode='lines',
        line=dict(color='rgba(0, 0, 128, 0.7)', width=2)
    ))
    
    # Previsioni
    fig_ml.add_trace(go.Scatter(
        x=future_normal['date'],
        y=future_normal['revenue_predicted'],
        name='Previsione (Normale)',
        mode='lines',
        line=dict(color='rgba(26, 118, 255, 1.0)', width=3)
    ))
    
    fig_ml.add_trace(go.Scatter(
        x=future_worst['date'],
        y=future_worst['revenue_predicted'],
        name='Previsione (Pessimistica)',
        mode='lines',
        line=dict(color='rgba(219, 64, 82, 0.7)', dash='dot')
    ))
    
    fig_ml.add_trace(go.Scatter(
        x=future_best['date'],
        y=future_best['revenue_predicted'],
        name='Previsione (Ottimistica)',
        mode='lines',
        line=dict(color='rgba(0, 128, 0, 0.7)', dash='dot')
    ))
    
    fig_ml.update_layout(
        title='Previsione Ricavi con LightGBM (Dati Mensili)',
        xaxis_title='Data',
        yaxis_title='Ricavi Mensili (€)',
        template='plotly_white',
        height=500
    )
    
    # 6. Confronto ricavi: deterministico vs ML (annuale)
    fig_compare = go.Figure()
    
    # Ricavi scenario normale (deterministico)
    fig_compare.add_trace(go.Bar(
        x=list(range(1, 6)),
        y=df_normal['revenue_total'],
        name='Deterministico - Normale',
        marker_color='rgba(26, 118, 255, 0.7)'
    ))
    
    # Ricavi scenario normale (ML)
    fig_compare.add_trace(go.Bar(
        x=annual_normal['year_num'],
        y=annual_normal['revenue_predicted'],
        name='ML - Normale',
        marker_color='rgba(26, 118, 255, 0.3)'
    ))
    
    # Ricavi scenario worst (deterministico)
    fig_compare.add_trace(go.Bar(
        x=list(range(1, 6)),
        y=df_worst['revenue_total'],
        name='Deterministico - Pessimistico',
        marker_color='rgba(219, 64, 82, 0.7)'
    ))
    
    # Ricavi scenario worst (ML)
    fig_compare.add_trace(go.Bar(
        x=annual_worst['year_num'],
        y=annual_worst['revenue_predicted'],
        name='ML - Pessimistico',
        marker_color='rgba(219, 64, 82, 0.3)'
    ))
    
    # Ricavi scenario best (deterministico)
    fig_compare.add_trace(go.Bar(
        x=list(range(1, 6)),
        y=df_best['revenue_total'],
        name='Deterministico - Ottimistico',
        marker_color='rgba(0, 128, 0, 0.7)'
    ))
    
    # Ricavi scenario best (ML)
    fig_compare.add_trace(go.Bar(
        x=annual_best['year_num'],
        y=annual_best['revenue_predicted'],
        name='ML - Ottimistico',
        marker_color='rgba(0, 128, 0, 0.3)'
    ))
    
    fig_compare.update_layout(
        title='Confronto Ricavi: Modello Deterministico vs LightGBM (Dati Annuali)',
        xaxis_title='Anno',
        yaxis_title='Ricavi Annuali (€)',
        template='plotly_white',
        height=500,
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    
    # 7. Grafico simulazione Monte Carlo
    fig_mc = go.Figure()
    
    # Plot delle proiezioni Monte Carlo (scenario normale)
    # Mostriamo solo 50 simulazioni per non appesantire il grafico
    sample_paths = mc_results['normal']['paths'][:, np.random.choice(params['num_simulations'], 50, replace=False)]
    
    for i in range(sample_paths.shape[1]):
        fig_mc.add_trace(go.Scatter(
            x=list(range(1, 6)),
            y=sample_paths[:, i],
            mode='lines',
            line=dict(color='rgba(26, 118, 255, 0.1)'),
            showlegend=False
        ))
    
    # Media, mediana e percentili
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['normal']['mean'],
        name='Media (Normale)',
        mode='lines+markers',
        line=dict(color='rgba(26, 118, 255, 1.0)', width=3),
        marker=dict(size=10)
    ))
    
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['normal']['p10'],
        name='10° Percentile (Normale)',
        mode='lines',
        line=dict(color='rgba(26, 118, 255, 0.7)', dash='dot')
    ))
    
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['normal']['p90'],
        name='90° Percentile (Normale)',
        mode='lines',
        line=dict(color='rgba(26, 118, 255, 0.7)', dash='dot'),
        fill='tonexty',
        fillcolor='rgba(26, 118, 255, 0.1)'
    ))
    
    # Scenario worst - solo media
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['worst']['mean'],
        name='Media (Pessimistico)',
        mode='lines+markers',
        line=dict(color='rgba(219, 64, 82, 0.7)', dash='dot'),
        marker=dict(size=8)
    ))
    
    # Scenario best - solo media
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['best']['mean'],
        name='Media (Ottimistico)',
        mode='lines+markers',
        line=dict(color='rgba(0, 128, 0, 0.7)', dash='dot'),
        marker=dict(size=8)
    ))
    
    # Confronto con proiezione deterministica
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=df_normal['revenue_total'],
        name='Deterministico (Normale)',
        mode='lines+markers',
        line=dict(color='black', width=2, dash='dashdot'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig_mc.update_layout(
        title='Simulazione Monte Carlo dei Ricavi (1000 simulazioni)',
        xaxis_title='Anno',
        yaxis_title='Ricavi Annuali (€)',
        template='plotly_white',
        height=600
    )
    
    # 8. Grafico del ROI per i tre scenari
    fig_roi = go.Figure()
    
    # Calcolo ROI
    roi_normal = (df_normal['cumulative_cashflow'].iloc[-1] + initial_investment) / initial_investment * 100 - 100
    roi_worst = (df_worst['cumulative_cashflow'].iloc[-1] + initial_investment) / initial_investment * 100 - 100
    roi_best = (df_best['cumulative_cashflow'].iloc[-1] + initial_investment) / initial_investment * 100 - 100
    
    fig_roi.add_trace(go.Bar(
        x=['Pessimistico', 'Normale', 'Ottimistico'],
        y=[roi_worst, roi_normal, roi_best],
        text=[f'{roi_worst:.1f}%', f'{roi_normal:.1f}%', f'{roi_best:.1f}%'],
        textposition='auto',
        marker_color=['rgba(219, 64, 82, 0.7)', 'rgba(26, 118, 255, 0.7)', 'rgba(0, 128, 0, 0.7)']
    ))
    
    fig_roi.update_layout(
        title='ROI a 5 anni per i tre scenari',
        xaxis_title='Scenario',
        yaxis_title='ROI (%)',
        template='plotly_white',
        height=500
    )
    
    return {
        'investment': fig_investment,
        'customers': fig_customers,
        'revenue_cost': fig_revenue_cost,
        'cashflow': fig_cashflow,
        'ml': fig_ml,
        'compare': fig_compare,
        'mc': fig_mc,
        'roi': fig_roi
    }

# 9. CREAZIONE DASHBOARD INTERATTIVA HTML
def create_dashboard():
    # Generazione grafici
    figures = create_plotly_figures()
    
    # Calcolo break-even
    breakeven = calculate_breakeven()
    
    # Preparazione HTML
    dashboard_html = get_interactive_plot()
    
    # Aggiunta sezione investimento iniziale
    dashboard_html += """
    <div style="margin-top: 20px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">1. Investimento Iniziale</h2>
            <p>Il capitale necessario per avviare la nostra fintech basata su chatbot AI è di <b>€{:,.0f}</b>, 
            suddiviso nelle seguenti categorie principali:</p>
        </div>
    """.format(initial_investment)
    
    # Grafico investimento
    dashboard_html += """
        <div id="investment-chart" style="width:100%;"></div>
        <script>
            var investment_fig = {};
            Plotly.newPlot('investment-chart', investment_fig.data, investment_fig.layout);
        </script>
    </div>
    """.format(figures['investment'].to_json())
    
    # Aggiunta sezione stima dei costi
    dashboard_html += """
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">2. Stima dei Costi</h2>
            <p>I costi operativi annuali variano nei diversi scenari previsionali. 
               Di seguito la ripartizione per categoria e l'evoluzione nel tempo:</p>
        </div>
        <div class="row">
            <div class="col-md-6">
                <h4>Costi Operativi Annuali (Anno 1)</h4>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Categoria</th>
                            <th>Importo (€)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Personale tecnico</td>
                            <td>{:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Personale supporto</td>
                            <td>{:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Personale conformità</td>
                            <td>{:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Cloud e hosting</td>
                            <td>{:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Manutenzione sistema</td>
                            <td>{:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Marketing</td>
                            <td>{:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Formazione</td>
                            <td>{:,.0f}</td>
                        </tr>
                        <tr>
                            <td><strong>Totale</strong></td>
                            <td><strong>{:,.0f}</strong></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div class="col-md-6">
                <h4>Evoluzione Costi vs Ricavi</h4>
                <div id="revenue-cost-chart" style="width:100%;"></div>
                <script>
                    var revenue_cost_fig = {};
                    Plotly.newPlot('revenue-cost-chart', revenue_cost_fig.data, revenue_cost_fig.layout);
                </script>
            </div>
        </div>
    </div>
    """.format(
        params['personale_tecnico'],
        params['personale_supporto'],
        params['personale_conformita'],
        params['cloud_hosting'],
        params['manutenzione_sistema'],
        params['marketing_annuale'],
        params['formazione'],
        params['personale_tecnico'] + params['personale_supporto'] + params['personale_conformita'] + 
        params['cloud_hosting'] + params['manutenzione_sistema'] + params['marketing_annuale'] + params['formazione'],
        figures['revenue_cost'].to_json()
    )
    
    # Aggiunta sezione evoluzione clienti
    dashboard_html += """
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">3. Evoluzione Base Clienti</h2>
            <p>La nostra strategia mira a raggiungere una base clienti significativa di giovani under 35, 
               con un mix ottimale tra clienti base e premium:</p>
        </div>
        <div id="customers-chart" style="width:100%;"></div>
        <script>
            var customers_fig = {};
            Plotly.newPlot('customers-chart', customers_fig.data, customers_fig.layout);
        </script>
    </div>
    """.format(figures['customers'].to_json())
    
    # Aggiunta sezione proiezioni finanziarie
    dashboard_html += """
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">4. Proiezioni Finanziarie a 5 Anni</h2>
            <p>Le proiezioni finanziarie mostrano un percorso di crescita solido nei tre scenari considerati.
               Il punto di break-even è previsto in {:.1f} anni nello scenario normale, 
               {:.1f} anni in quello pessimistico e {:.1f} anni in quello ottimistico.</p>
        </div>
        <div id="cashflow-chart" style="width:100%;"></div>
        <script>
            var cashflow_fig = {};
            Plotly.newPlot('cashflow-chart', cashflow_fig.data, cashflow_fig.layout);
        </script>
    </div>
    """.format(
        breakeven['normal'] if breakeven['normal'] is not None else float('inf'),
        breakeven['worst'] if breakeven['worst'] is not None else float('inf'),
        breakeven['best'] if breakeven['best'] is not None else float('inf'),
        figures['cashflow'].to_json()
    )
    
    # Aggiunta sezione previsione ML
    dashboard_html += """
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">5. Previsione Ricavi con Machine Learning</h2>
            <p>Utilizzando l'algoritmo LightGBM, abbiamo generato previsioni di ricavi basate su dati storici sintetici.
               Il modello cattura trend e stagionalità nei ricavi:</p>
        </div>
        <div id="ml-chart" style="width:100%;"></div>
        <script>
            var ml_fig = {};
            Plotly.newPlot('ml-chart', ml_fig.data, ml_fig.layout);
        </script>
        <div style="margin-top: 20px;">
            <div id="compare-chart" style="width:100%;"></div>
            <script>
                var compare_fig = {};
                Plotly.newPlot('compare-chart', compare_fig.data, compare_fig.layout);
            </script>
        </div>
    </div>
    """.format(figures['ml'].to_json(), figures['compare'].to_json())
    
    # Aggiunta sezione Monte Carlo
    dashboard_html += """
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">6. Simulazione Monte Carlo</h2>
            <p>La simulazione Monte Carlo con 1000 iterazioni permette di valutare la distribuzione probabilistica
               dei risultati finanziari, considerando la volatilità del mercato e l'incertezza delle proiezioni:</p>
        </div>
        <div id="mc-chart" style="width:100%;"></div>
        <script>
            var mc_fig = {};
            Plotly.newPlot('mc-chart', mc_fig.data, mc_fig.layout);
        </script>
    </div>
    """.format(figures['mc'].to_json())
    
    # Aggiunta sezione ROI
    dashboard_html += """
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">7. Return on Investment (ROI)</h2>
            <p>Il ROI atteso a 5 anni varia significativamente tra gli scenari, 
               evidenziando il potenziale di rendimento dell'investimento:</p>
        </div>
        <div id="roi-chart" style="width:100%;"></div>
        <script>
            var roi_fig = {};
            Plotly.newPlot('roi-chart', roi_fig.data, roi_fig.layout);
        </script>
    </div>
    """.format(figures['roi'].to_json())
    
    # Aggiungi conclusioni
    dashboard_html += """
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">8. Conclusioni</h2>
            <p>Dall'analisi emergono i seguenti punti chiave:</p>
            <ul>
                <li>L'investimento iniziale di <b>€{:,.0f}</b> è significativo ma proporzionato al mercato fintech</li>
                <li>Il break-even si prevede in {:.1f} anni nello scenario normale</li>
                <li>Il ROI a 5 anni risulta del {:.1f}% nello scenario normale</li>
                <li>La crescita della base clienti under 35 è il driver principale della redditività</li>
                <li>La simulazione Monte Carlo mostra una probabilità del 90% di ottenere ricavi superiori a 
                    €{:,.0f} nel quinto anno</li>
            </ul>
            <p>Il modello di business della fintech mostra solide potenzialità di crescita, con risultati 
               particolarmente positivi nel target under 35, caratterizzato da maggiore propensione all'adozione
               di soluzioni basate su AI e chatbot.</p>
        </div>
    </div>
    """.format(
        initial_investment,
        breakeven['normal'] if breakeven['normal'] is not None else float('inf'),
        (df_normal['cumulative_cashflow'].iloc[-1] + initial_investment) / initial_investment * 100 - 100,
        figures['mc'].data[2].y[-1]  # 10° percentile del quinto anno
    )
    
    return dashboard_html

# Esecuzione principale
def main():
    # Calcola l'investimento iniziale
    print(f"Investimento iniziale: €{initial_investment:,.2f}")
    
    # Simula i flussi finanziari
    df_normal = simulate_financials('normal')
    df_worst = simulate_financials('worst')
    df_best = simulate_financials('best')
    
    # Stampa le previsioni a 5 anni
    print("\nPrevisioni finanziarie a 5 anni (scenario normale):")
    print(df_normal[['year', 'customers', 'revenue_total', 'cost_total', 'ebitda', 'cumulative_cashflow']])
    
    # Calcola il break-even point
    breakeven = calculate_breakeven()
    print(f"\nBreak-even point (anni):")
    print(f"Scenario normale: {breakeven['normal']:.2f}")
    print(f"Scenario pessimistico: {breakeven['worst'] if breakeven['worst'] is not None else 'Non raggiunto in 5 anni'}")
    print(f"Scenario ottimistico: {breakeven['best']:.2f}")
    
    # Crea la dashboard HTML
    dashboard_html = create_dashboard()
    
    return dashboard_html

# Entry point
dashboard_html = main()

# Output finale con Bootstrap e Plotly inclusi
html_output = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analisi del Fabbisogno Finanziario - Fintech Chatbot AI</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        table {
            width: 100%;
            margin-bottom: 1rem;
            border-collapse: collapse;
        }
        table td, table th {
            padding: 0.75rem;
            border: 1px solid #dee2e6;
        }
        table th {
            background-color: #f8f9fa;
        }
    </style>
</head>
<body>
    <div class="container">
        """ + dashboard_html + """
    </div>
    
    <!-- Bootstrap JS Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Salva l'HTML in un file
with open('analisi_finanziaria.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

# Apri automaticamente il file HTML nel browser predefinito
import webbrowser
webbrowser.open('analisi_finanziaria.html')

print("\nDashboard HTML salvata e aperta nel browser.")