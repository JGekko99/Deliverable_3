import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from numba import jit, prange
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
import io
import base64
from scipy.stats import norm
import webbrowser

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
            Analisi deterministica e simulazioni Monte Carlo per 5 anni
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
    'num_simulations': 1000000,          # Numero di simulazioni Monte Carlo
    'volatility': 0.25,               # Volatilità per simulazione Monte Carlo
    'volatility_worst': 0.35,         # Volatilità aumentata per scenario pessimistico
    'volatility_best': 0.20,          # Volatilità ridotta per scenario ottimistico
    'confidence_intervals': [0.10, 0.25, 0.50, 0.75, 0.90],  # Intervalli di confidenza per MC
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

# SIMULAZIONE MONTE CARLO CON NUMBA - Versione potenziata
@jit(nopython=True, parallel=True)
def monte_carlo_simulation(starting_capital, years, num_sims, growth_rates, volatilities, seed=42):
    """
    Simulazione Monte Carlo accelerata con Numba e parallelizzata
    """
    np.random.seed(seed)
    paths = np.zeros((years, num_sims))
    
    for sim in prange(num_sims):
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

# Funzione estesa per simulazione Monte Carlo con analisi probabilistica
def run_advanced_monte_carlo():
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
    volatilities_normal = [base_vol * (1 - 0.1*i) for i in range(years)]
    
    # Volatilità differenziata per scenario
    volatilities_worst = [params['volatility_worst'] * (1 - 0.08*i) for i in range(years)]
    volatilities_best = [params['volatility_best'] * (1 - 0.12*i) for i in range(years)]
    
    # Capitale iniziale (primo anno di ricavi dal modello deterministico)
    starting_capital_normal = df_normal['revenue_total'][0]
    starting_capital_worst = df_worst['revenue_total'][0]
    starting_capital_best = df_best['revenue_total'][0]
    
    # Esecuzione simulazioni per i tre scenari
    mc_paths_normal = monte_carlo_simulation(
        starting_capital_normal, years, num_sims, 
        np.array(normal_growth_rates), np.array(volatilities_normal), seed=42
    )
    
    mc_paths_worst = monte_carlo_simulation(
        starting_capital_worst, years, num_sims, 
        np.array(worst_growth_rates), np.array(volatilities_worst), seed=42
    )
    
    mc_paths_best = monte_carlo_simulation(
        starting_capital_best, years, num_sims, 
        np.array(best_growth_rates), np.array(volatilities_best), seed=42
    )
    
    # Calcolo statistiche per vari intervalli di confidenza
    ci_results = {}
    for scenario_name, paths in [('normal', mc_paths_normal), 
                                ('worst', mc_paths_worst), 
                                ('best', mc_paths_best)]:
        ci_results[scenario_name] = {
            'mean': np.mean(paths, axis=1),
            'median': np.median(paths, axis=1),
            'paths': paths
        }
        
        # Aggiungi percentili per vari intervalli di confidenza
        for ci in params['confidence_intervals']:
            lower_key = f'p{int(ci*100)}'
            upper_key = f'p{int((1-ci)*100)}'
            ci_results[scenario_name][lower_key] = np.percentile(paths, ci*100, axis=1)
            ci_results[scenario_name][upper_key] = np.percentile(paths, (1-ci)*100, axis=1)
    
    # Analisi di probabilità di successo (raggiungimento di target specifici)
    success_analysis = {}
    
    # Target di ricavi per anno 5
    revenue_targets = [30000000, 50000000, 70000000, 100000000]
    
    for scenario_name, paths in [('normal', mc_paths_normal), 
                               ('worst', mc_paths_worst), 
                               ('best', mc_paths_best)]:
        success_analysis[scenario_name] = {}
        
        # Probabilità di superare i target di ricavi all'anno 5
        for target in revenue_targets:
            prob = np.mean(paths[years-1, :] > target) * 100
            # Correzione: Usa int() per evitare decimali nelle chiavi
            success_analysis[scenario_name][f'prob_exceed_{int(target/1000000)}M'] = prob
        
        # Analisi ROI
        initial_invest = initial_investment
        roi_values = (paths[years-1, :] - starting_capital_normal) / initial_invest * 100
        
        # Distribuzione ROI al quinto anno
        success_analysis[scenario_name]['roi_mean'] = np.mean(roi_values)
        success_analysis[scenario_name]['roi_median'] = np.median(roi_values)
        success_analysis[scenario_name]['roi_p10'] = np.percentile(roi_values, 10)
        success_analysis[scenario_name]['roi_p25'] = np.percentile(roi_values, 25)
        success_analysis[scenario_name]['roi_p75'] = np.percentile(roi_values, 75)
        success_analysis[scenario_name]['roi_p90'] = np.percentile(roi_values, 90)
        
        # Probabilità di ottenere ROI positivo
        success_analysis[scenario_name]['prob_positive_roi'] = np.mean(roi_values > 0) * 100
        
        # Probabilità di ottenere ROI > 100%
        success_analysis[scenario_name]['prob_roi_100'] = np.mean(roi_values > 100) * 100
        
        # Probabilità di ottenere ROI > 200%
        success_analysis[scenario_name]['prob_roi_200'] = np.mean(roi_values > 200) * 100
        
        # Probabilità di ottenere ROI > 500%
        success_analysis[scenario_name]['prob_roi_500'] = np.mean(roi_values > 500) * 100
    
    # Analisi di sensitività - come cambiano i risultati rispetto a parametri chiave
    sensitivity = {}
    
    # Calcoliamo il valore medio finale per diverse volatilità
    vol_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    sensitivity['volatility'] = {}
    
    for vol in vol_range:
        volatilities_test = [vol * (1 - 0.1*i) for i in range(years)]
        
        paths = monte_carlo_simulation(
            starting_capital_normal, years, num_sims//2,  # riduciamo il numero di simulazioni per velocità
            np.array(normal_growth_rates), np.array(volatilities_test), seed=42
        )
        
        sensitivity['volatility'][vol] = {
            'mean_final': np.mean(paths[years-1, :]),
            'p10_final': np.percentile(paths[years-1, :], 10),
            'p90_final': np.percentile(paths[years-1, :], 90)
        }
    
    # Calcoliamo l'impatto di diversi tassi di crescita
    growth_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    sensitivity['growth'] = {}
    
    for factor in growth_factors:
        modified_growth_rates = [rate * factor for rate in normal_growth_rates]
        
        paths = monte_carlo_simulation(
            starting_capital_normal, years, num_sims//2,
            np.array(modified_growth_rates), np.array(volatilities_normal), seed=42
        )
        
        sensitivity['growth'][factor] = {
            'mean_final': np.mean(paths[years-1, :]),
            'p10_final': np.percentile(paths[years-1, :], 10),
            'p90_final': np.percentile(paths[years-1, :], 90)
        }
    
    return {
        'ci_results': ci_results,
        'success_analysis': success_analysis,
        'sensitivity': sensitivity
    }

# CALCOLO DEL BREAK-EVEN POINT PER OGNI SCENARIO
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

# CREAZIONE GRAFICI INTERATTIVI CON PLOTLY
def create_plotly_figures():
    # Simulazione Monte Carlo potenziata
    mc_results = run_advanced_monte_carlo()
    
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
    
    # 5. Grafico della simulazione Monte Carlo potenziata
    fig_mc = go.Figure()
    
    # Plot delle proiezioni Monte Carlo (scenario normale)
    # Mostriamo solo 50 simulazioni per non appesantire il grafico
    sample_paths = mc_results['ci_results']['normal']['paths'][:, np.random.choice(params['num_simulations'], 50, replace=False)]
    
    for i in range(sample_paths.shape[1]):
        fig_mc.add_trace(go.Scatter(
            x=list(range(1, 6)),
            y=sample_paths[:, i],
            mode='lines',
            line=dict(color='rgba(26, 118, 255, 0.1)'),
            showlegend=False
        ))
    
    # Media, mediana e percentili per scenario normale
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['ci_results']['normal']['mean'],
        name='Media (Normale)',
        mode='lines+markers',
        line=dict(color='rgba(26, 118, 255, 1.0)', width=3),
        marker=dict(size=10)
    ))
    
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['ci_results']['normal']['p10'],
        name='10° Percentile (Normale)',
        mode='lines',
        line=dict(color='rgba(26, 118, 255, 0.7)', dash='dot')
    ))
    
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['ci_results']['normal']['p90'],
        name='90° Percentile (Normale)',
        mode='lines',
        line=dict(color='rgba(26, 118, 255, 0.7)', dash='dot'),
        fill='tonexty',
        fillcolor='rgba(26, 118, 255, 0.1)'
    ))
    
    # Media per gli scenari worst e best
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['ci_results']['worst']['mean'],
        name='Media (Pessimistico)',
        mode='lines+markers',
        line=dict(color='rgba(219, 64, 82, 0.7)', dash='dot'),
        marker=dict(size=8)
    ))
    
    fig_mc.add_trace(go.Scatter(
        x=list(range(1, 6)),
        y=mc_results['ci_results']['best']['mean'],
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
        title=f'Simulazione Monte Carlo dei Ricavi ({params["num_simulations"]} simulazioni)',
        xaxis_title='Anno',
        yaxis_title='Ricavi Annuali (€)',
        template='plotly_white',
        height=600
    )
    
    # 6. Grafico della distribuzione dei risultati al quinto anno
    fig_dist = go.Figure()
    
    # Kernel Density Estimation per i risultati del quinto anno
    year5_normal = mc_results['ci_results']['normal']['paths'][4, :]
    year5_worst = mc_results['ci_results']['worst']['paths'][4, :]
    year5_best = mc_results['ci_results']['best']['paths'][4, :]
    
    # Utilizziamo un istogramma con sovrapposizione KDE per i tre scenari
    fig_dist.add_trace(go.Histogram(
        x=year5_normal,
        name='Scenario Normale',
        histnorm='probability density',
        marker_color='rgba(26, 118, 255, 0.6)',
        nbinsx=50
    ))
    
    fig_dist.add_trace(go.Histogram(
        x=year5_worst,
        name='Scenario Pessimistico',
        histnorm='probability density',
        marker_color='rgba(219, 64, 82, 0.6)',
        nbinsx=50
    ))
    
    fig_dist.add_trace(go.Histogram(
        x=year5_best,
        name='Scenario Ottimistico',
        histnorm='probability density',
        marker_color='rgba(0, 128, 0, 0.6)',
        nbinsx=50
    ))
    
    # Aggiungiamo linee verticali per i valori target
    revenue_targets = [30000000, 50000000, 70000000, 100000000]
    for target in revenue_targets:
        fig_dist.add_shape(
            type="line",
            x0=target, y0=0,
            x1=target, y1=0.00000007,  # altezza adattata in base alla scala degli istogrammi
            line=dict(color="red", width=2, dash="dash"),
        )
        fig_dist.add_annotation(
            x=target,
            y=0.00000008,
            text=f"€{target/1000000:.0f}M",
            showarrow=False,
            yshift=10
        )
    
    fig_dist.update_layout(
        title='Distribuzione Probabilistica dei Ricavi al Quinto Anno',
        xaxis_title='Ricavi (€)',
        yaxis_title='Densità di Probabilità',
        template='plotly_white',
        height=500,
        barmode='overlay',
        bargap=0.1
    )
    
    # 7. Analisi di sensitività
    fig_sensitivity = make_subplots(rows=1, cols=2, 
                                   subplot_titles=('Impatto della Volatilità', 
                                                   'Impatto del Tasso di Crescita'))
    
    # Grafico dell'impatto della volatilità
    vol_values = list(mc_results['sensitivity']['volatility'].keys())
    mean_values = [mc_results['sensitivity']['volatility'][k]['mean_final'] for k in vol_values]
    p10_values = [mc_results['sensitivity']['volatility'][k]['p10_final'] for k in vol_values]
    p90_values = [mc_results['sensitivity']['volatility'][k]['p90_final'] for k in vol_values]
    
    fig_sensitivity.add_trace(
        go.Scatter(x=vol_values, y=mean_values, mode='lines+markers', 
                  name='Media', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig_sensitivity.add_trace(
        go.Scatter(x=vol_values, y=p10_values, mode='lines+markers', 
                  name='P10', line=dict(color='red', width=2, dash='dot')),
        row=1, col=1
    )
    
    fig_sensitivity.add_trace(
        go.Scatter(x=vol_values, y=p90_values, mode='lines+markers', 
                  name='P90', line=dict(color='green', width=2, dash='dot')),
        row=1, col=1
    )
    
    # Grafico dell'impatto del tasso di crescita
    growth_values = list(mc_results['sensitivity']['growth'].keys())
    mean_values = [mc_results['sensitivity']['growth'][k]['mean_final'] for k in growth_values]
    p10_values = [mc_results['sensitivity']['growth'][k]['p10_final'] for k in growth_values]
    p90_values = [mc_results['sensitivity']['growth'][k]['p90_final'] for k in growth_values]
    
    fig_sensitivity.add_trace(
        go.Scatter(x=growth_values, y=mean_values, mode='lines+markers', 
                  name='Media', line=dict(color='blue', width=2)),
        row=1, col=2
    )
    
    fig_sensitivity.add_trace(
        go.Scatter(x=growth_values, y=p10_values, mode='lines+markers', 
                  name='P10', line=dict(color='red', width=2, dash='dot')),
        row=1, col=2
    )
    
    fig_sensitivity.add_trace(
        go.Scatter(x=growth_values, y=p90_values, mode='lines+markers', 
                  name='P90', line=dict(color='green', width=2, dash='dot')),
        row=1, col=2
    )
    
    fig_sensitivity.update_xaxes(title_text="Volatilità", row=1, col=1)
    fig_sensitivity.update_xaxes(title_text="Fattore di Crescita", row=1, col=2)
    fig_sensitivity.update_yaxes(title_text="Ricavi Anno 5 (€)", row=1, col=1)
    fig_sensitivity.update_yaxes(title_text="Ricavi Anno 5 (€)", row=1, col=2)
    
    fig_sensitivity.update_layout(
        title='Analisi di Sensitività dei Parametri Chiave',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    # 8. Grafico del ROI per i tre scenari
    fig_roi = go.Figure()
    
    # Calcolo ROI deterministico
    roi_normal = (df_normal['cumulative_cashflow'].iloc[-1] + initial_investment) / initial_investment * 100 - 100
    roi_worst = (df_worst['cumulative_cashflow'].iloc[-1] + initial_investment) / initial_investment * 100 - 100
    roi_best = (df_best['cumulative_cashflow'].iloc[-1] + initial_investment) / initial_investment * 100 - 100
    
    # Dati ROI da Monte Carlo (mediana e quartili)
    roi_mc_normal = mc_results['success_analysis']['normal']['roi_median']
    roi_mc_worst = mc_results['success_analysis']['worst']['roi_median']
    roi_mc_best = mc_results['success_analysis']['best']['roi_median']
    
    # ROI deterministico
    fig_roi.add_trace(go.Bar(
        x=['Pessimistico', 'Normale', 'Ottimistico'],
        y=[roi_worst, roi_normal, roi_best],
        text=[f'{roi_worst:.1f}%', f'{roi_normal:.1f}%', f'{roi_best:.1f}%'],
        textposition='auto',
        name='ROI Deterministico',
        marker_color=['rgba(219, 64, 82, 0.7)', 'rgba(26, 118, 255, 0.7)', 'rgba(0, 128, 0, 0.7)']
    ))
    
    # ROI da Monte Carlo (mediana)
    fig_roi.add_trace(go.Bar(
        x=['Pessimistico', 'Normale', 'Ottimistico'],
        y=[roi_mc_worst, roi_mc_normal, roi_mc_best],
        text=[f'{roi_mc_worst:.1f}%', f'{roi_mc_normal:.1f}%', f'{roi_mc_best:.1f}%'],
        textposition='auto',
        name='ROI Monte Carlo (Mediana)',
        marker_color=['rgba(219, 64, 82, 0.3)', 'rgba(26, 118, 255, 0.3)', 'rgba(0, 128, 0, 0.3)']
    ))
    
    # Probabilità di ROI > 100% (raddoppio del capitale)
    prob_roi_100_normal = mc_results['success_analysis']['normal']['prob_roi_100']
    prob_roi_100_worst = mc_results['success_analysis']['worst']['prob_roi_100']
    prob_roi_100_best = mc_results['success_analysis']['best']['prob_roi_100']
    
    # Aggiungiamo annotazioni per le probabilità
    fig_roi.add_annotation(
        x='Normale', y=roi_mc_normal + 50,
        text=f"P(ROI>100%): {prob_roi_100_normal:.1f}%",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig_roi.add_annotation(
        x='Pessimistico', y=roi_mc_worst + 50,
        text=f"P(ROI>100%): {prob_roi_100_worst:.1f}%",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig_roi.add_annotation(
        x='Ottimistico', y=roi_mc_best + 50,
        text=f"P(ROI>100%): {prob_roi_100_best:.1f}%",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig_roi.update_layout(
        title='ROI a 5 anni: Confronto tra Modello Deterministico e Simulazione Monte Carlo',
        xaxis_title='Scenario',
        yaxis_title='ROI (%)',
        template='plotly_white',
        height=500,
        barmode='group'
    )
    
    # 9. Tabella di probabilità per vari target di successo
    def create_probability_table():
        # Formattazione per la tabella HTML
        html = """
        <div class="table-responsive">
            <table class="table table-striped table-hover">
                <thead class="thead-dark">
                    <tr>
                        <th>Target</th>
                        <th>Scenario Pessimistico</th>
                        <th>Scenario Normale</th>
                        <th>Scenario Ottimistico</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Riga per la probabilità di ROI positivo
        html += f"""
            <tr>
                <td><strong>ROI Positivo</strong></td>
                <td>{mc_results['success_analysis']['worst']['prob_positive_roi']:.1f}%</td>
                <td>{mc_results['success_analysis']['normal']['prob_positive_roi']:.1f}%</td>
                <td>{mc_results['success_analysis']['best']['prob_positive_roi']:.1f}%</td>
            </tr>
        """
        
        # Righe per la probabilità di ROI > 100%, 200%, 500%
        roi_targets = [
            ('ROI > 100%', 'prob_roi_100'),
            ('ROI > 200%', 'prob_roi_200'),
            ('ROI > 500%', 'prob_roi_500')
        ]
        
        for label, key in roi_targets:
            html += f"""
                <tr>
                    <td><strong>{label}</strong></td>
                    <td>{mc_results['success_analysis']['worst'][key]:.1f}%</td>
                    <td>{mc_results['success_analysis']['normal'][key]:.1f}%</td>
                    <td>{mc_results['success_analysis']['best'][key]:.1f}%</td>
                </tr>
            """
        
        # Righe per la probabilità di superare vari target di ricavi
        revenue_targets = [30, 50, 70, 100]
        
        for target in revenue_targets:
            key = f'prob_exceed_{target}M'
            html += f"""
                <tr>
                    <td><strong>Ricavi > €{target}M (Anno 5)</strong></td>
                    <td>{mc_results['success_analysis']['worst'][key]:.1f}%</td>
                    <td>{mc_results['success_analysis']['normal'][key]:.1f}%</td>
                    <td>{mc_results['success_analysis']['best'][key]:.1f}%</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    return {
        'investment': fig_investment,
        'customers': fig_customers,
        'revenue_cost': fig_revenue_cost,
        'cashflow': fig_cashflow,
        'mc': fig_mc,
        'dist': fig_dist,
        'sensitivity': fig_sensitivity,
        'roi': fig_roi,
        'probability_table': create_probability_table(),
        'mc_results': mc_results
    }

# CREAZIONE DASHBOARD INTERATTIVA HTML
def create_dashboard():
    # Generazione grafici
    figures = create_plotly_figures()
    
    # Calcolo break-even
    breakeven = calculate_breakeven()
    
    # Preparazione HTML
    dashboard_html = get_interactive_plot()
    
    # Aggiunta sezione investimento iniziale
    dashboard_html += f"""
    <div style="margin-top: 20px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">1. Investimento Iniziale</h2>
            <p>Il capitale necessario per avviare la nostra fintech basata su chatbot AI è di <b>€{initial_investment:,.0f}</b>, 
            suddiviso nelle seguenti categorie principali:</p>
        </div>
    """
    
    # Grafico investimento
    dashboard_html += f"""
        <div id="investment-chart" style="width:100%;"></div>
        <script>
            var investment_fig = {figures['investment'].to_json()};
            Plotly.newPlot('investment-chart', investment_fig.data, investment_fig.layout);
        </script>
    </div>
    """
    
    # Aggiunta sezione stima dei costi
    costi_totali = (params['personale_tecnico'] + params['personale_supporto'] + 
                   params['personale_conformita'] + params['cloud_hosting'] + 
                   params['manutenzione_sistema'] + params['marketing_annuale'] + 
                   params['formazione'])
    
    dashboard_html += f"""
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
                            <td>{params['personale_tecnico']:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Personale supporto</td>
                            <td>{params['personale_supporto']:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Personale conformità</td>
                            <td>{params['personale_conformita']:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Cloud e hosting</td>
                            <td>{params['cloud_hosting']:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Manutenzione sistema</td>
                            <td>{params['manutenzione_sistema']:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Marketing</td>
                            <td>{params['marketing_annuale']:,.0f}</td>
                        </tr>
                        <tr>
                            <td>Formazione</td>
                            <td>{params['formazione']:,.0f}</td>
                        </tr>
                        <tr>
                            <td><strong>Totale</strong></td>
                            <td><strong>{costi_totali:,.0f}</strong></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div class="col-md-6">
                <h4>Evoluzione Costi vs Ricavi</h4>
                <div id="revenue-cost-chart" style="width:100%;"></div>
                <script>
                    var revenue_cost_fig = {figures['revenue_cost'].to_json()};
                    Plotly.newPlot('revenue-cost-chart', revenue_cost_fig.data, revenue_cost_fig.layout);
                </script>
            </div>
        </div>
    </div>
    """
    
    # Aggiunta sezione evoluzione clienti
    dashboard_html += f"""
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">3. Evoluzione Base Clienti</h2>
            <p>La nostra strategia mira a raggiungere una base clienti significativa di giovani under 35, 
               con un mix ottimale tra clienti base e premium:</p>
        </div>
        <div id="customers-chart" style="width:100%;"></div>
        <script>
            var customers_fig = {figures['customers'].to_json()};
            Plotly.newPlot('customers-chart', customers_fig.data, customers_fig.layout);
        </script>
    </div>
    """
    
    # Aggiunta sezione proiezioni finanziarie
    be_normal = breakeven['normal'] if breakeven['normal'] is not None else float('inf')
    be_worst = breakeven['worst'] if breakeven['worst'] is not None else float('inf')
    be_best = breakeven['best'] if breakeven['best'] is not None else float('inf')
    
    dashboard_html += f"""
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">4. Proiezioni Finanziarie a 5 Anni</h2>
            <p>Le proiezioni finanziarie mostrano un percorso di crescita solido nei tre scenari considerati.
               Il punto di break-even è previsto in {be_normal:.1f} anni nello scenario normale, 
               {be_worst:.1f} anni in quello pessimistico e {be_best:.1f} anni in quello ottimistico.</p>
        </div>
        <div id="cashflow-chart" style="width:100%;"></div>
        <script>
            var cashflow_fig = {figures['cashflow'].to_json()};
            Plotly.newPlot('cashflow-chart', cashflow_fig.data, cashflow_fig.layout);
        </script>
    </div>
    """
    
    # Aggiunta sezione Monte Carlo - USANDO F-STRING PER EVITARE PROBLEMI DI FORMATTAZIONE
    dashboard_html += f"""
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">5. Simulazione Monte Carlo</h2>
            <p>La simulazione Monte Carlo con {params['num_simulations']} iterazioni permette di valutare la distribuzione probabilistica
               dei risultati finanziari, considerando la volatilità del mercato e l'incertezza delle proiezioni:</p>
        </div>
        <div id="mc-chart" style="width:100%;"></div>
        <script>
            var mc_fig = {figures['mc'].to_json()};
            Plotly.newPlot('mc-chart', mc_fig.data, mc_fig.layout);
        </script>
        
        <div style="margin-top: 30px;">
            <h4>Distribuzione Probabilistica dei Risultati (Anno 5)</h4>
            <div id="dist-chart" style="width:100%;"></div>
            <script>
                var dist_fig = {figures['dist'].to_json()};
                Plotly.newPlot('dist-chart', dist_fig.data, dist_fig.layout);
            </script>
        </div>
    </div>
    """
    
    # Aggiunta sezione Analisi di Sensitività
    dashboard_html += f"""
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">6. Analisi di Sensitività</h2>
            <p>L'analisi di sensitività mostra come i risultati finanziari variano al cambiare dei parametri chiave
               come volatilità di mercato e tassi di crescita:</p>
        </div>
        <div id="sensitivity-chart" style="width:100%;"></div>
        <script>
            var sensitivity_fig = {figures['sensitivity'].to_json()};
            Plotly.newPlot('sensitivity-chart', sensitivity_fig.data, sensitivity_fig.layout);
        </script>
    </div>
    """
    
    # Aggiunta sezione ROI
    dashboard_html += f"""
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">7. Return on Investment (ROI)</h2>
            <p>Il ROI atteso a 5 anni varia significativamente tra gli scenari, 
               evidenziando il potenziale di rendimento dell'investimento:</p>
        </div>
        <div id="roi-chart" style="width:100%;"></div>
        <script>
            var roi_fig = {figures['roi'].to_json()};
            Plotly.newPlot('roi-chart', roi_fig.data, roi_fig.layout);
        </script>
        
        <div style="margin-top: 30px;">
            <h4>Tabella di Probabilità</h4>
            {figures['probability_table']}
        </div>
    </div>
    """
    
    # Aggiungi conclusioni
    roi_deterministico = (df_normal['cumulative_cashflow'].iloc[-1] + initial_investment) / initial_investment * 100 - 100
    roi_mc = figures['mc_results']['success_analysis']['normal']['roi_median']
    prob_roi_100 = figures['mc_results']['success_analysis']['normal']['prob_roi_100']
    prob_exceed_50M = figures['mc_results']['success_analysis']['normal']['prob_exceed_50M']
    
    dashboard_html += f"""
    <div style="margin-top: 40px; margin-bottom: 40px;">
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h2 style="color:#2c3e50;">8. Conclusioni</h2>
            <p>Dall'analisi emergono i seguenti punti chiave:</p>
            <ul>
                <li>L'investimento iniziale di <b>€{initial_investment:,.0f}</b> è significativo ma proporzionato al mercato fintech</li>
                <li>Il break-even si prevede in {be_normal:.1f} anni nello scenario normale</li>
                <li>Il ROI a 5 anni risulta del {roi_deterministico:.1f}% nello scenario normale (deterministico) e del {roi_mc:.1f}% (mediana Monte Carlo)</li>
                <li>La crescita della base clienti under 35 è il driver principale della redditività</li>
                <li>La simulazione Monte Carlo mostra una probabilità del {prob_roi_100:.1f}% di ottenere un ROI superiore al 100%</li>
                <li>La probabilità di superare €50M di ricavi al quinto anno è del {prob_exceed_50M:.1f}%</li>
            </ul>
            <p>Il modello di business della fintech mostra solide potenzialità di crescita, con risultati 
               particolarmente positivi nel target under 35, caratterizzato da maggiore propensione all'adozione
               di soluzioni basate su AI e chatbot.</p>
        </div>
    </div>
    """
    
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
        .table-responsive {
            overflow-x: auto;
        }
        .thead-dark th {
            background-color: #343a40;
            color: #fff;
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
with open('analisi_finanziaria2.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

# Apri automaticamente il file HTML nel browser predefinito
webbrowser.open('analisi_finanziaria2.html')

print("\nDashboard HTML salvata e aperta nel browser.")