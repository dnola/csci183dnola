__author__ = 'davidnola'

import bokeh.models
from bokeh.models.actions import Callback

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
from bokeh.plotting import figure, output_file, show
import bokeh
import bokeh.models
from bokeh.models import ColumnDataSource, Slider


def perform_regression():
    us_state_abbrev = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'DC',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY',
    }

    def pickle_heat_deaths():
        heat_deaths = []

        for i in range(2000,2005):
            print(i)
            df = pd.read_csv('mort%s.csv'%i)
            df = df[df['ucod'].str.contains('X30')]
            print(df.head())
            heat_deaths.append(df)

        pickle.dump(heat_deaths, open('heat_deaths.pkl','wb'))

    #pickle_heat_deaths()

    heat_deaths = pickle.load(open('heat_deaths.pkl','rb'))

    print(list(heat_deaths[0].columns))

    counts = {}


    for v in us_state_abbrev.values():
        counts[v]=0

    num_map = sorted(us_state_abbrev.keys())

    k_to_n = {}

    for i,n in enumerate(num_map):
        k_to_n[us_state_abbrev[n]] = i+1
    print(k_to_n)
    for k in counts.keys():
        for df in heat_deaths:
            try:
                counts[k]+=len(df[df['stateoc'].str.contains(k)])
            except:
                counts[k]+=len(df[df['stateoc']==k_to_n[k]])

            pass
            # counts[k]+=len(df[df['stateoc'].str.contains(k)]
            # print(df['stateoc'].unique())
            # print(df[df['stateoc'].str.contains(k)])

    pops_df = pd.read_csv('state_pop.csv')
    print(pops_df.head())
    pops = {}
    for st in us_state_abbrev.keys():
        if 'District of Columbia' in st:
            pops[us_state_abbrev[st]] = 567754
            continue

        o4 = pops_df[pops_df['State']==st]['2004']
        o0 = pops_df[pops_df['State']==st]['2000']
        avg = float((o4+o0)/2.0)
        pops[us_state_abbrev[st]] = float(avg)


    rates = {} # In Micromorts
    for k,v in counts.items():
        rates[k] = (counts[k] / (5* pops[k]))*10**6

    del(rates['DC'])

    descending = list(reversed(sorted(rates, key=rates.get)))
    print(rates)
    print(descending)

    temps_df = pd.read_csv('state_temp.csv')
    temps = {}
    for st in us_state_abbrev.keys():
        if 'District of Columbia' in st:
            continue

        temps[us_state_abbrev[st]] = float(temps_df[temps_df['State']==st]['F'])
    print(temps)

    temp_data = []
    death_data = []

    for k in rates.keys():
        death_data.append(rates[k])
        temp_data.append(temps[k])
    return (temp_data, death_data)


# plt.scatter(temp_data,death_data)
#########################################################################################
(temp_data, death_data) = perform_regression()

temp_data = (np.array(temp_data)-32)/1.8

from scipy.optimize import curve_fit

def func(x, a,b):
    return np.exp(a*x)-b

popt, pcov = curve_fit(func, temp_data, death_data, [.1,1])
print(popt)

x= np.linspace(-5,25,10000)

output_file("HeatDeathRegression2.html", title="Odds of Dying to Heat by State")

p = figure(tools='',width=700, height=350,title="State Tempurature vs Odds of Dying to Heat", x_axis_label='Average Tempurature in State (C)', y_axis_label='Heat Exposure (in Micromorts)')
p.scatter(temp_data, death_data, line_width=.2)
p.line(x, func(x,*popt), legend="Line of Best Fit", line_width=.5)
p.toolbar_location = None



global_rate = 30
carbon_rate = .029
carbon_decay = 0.0005
absorb_rate = .46
absorb_decay = .01
net_carbon = [(global_rate*(1+max(carbon_rate-carbon_decay*n,-1.35))**n)*min(1, (absorb_rate+absorb_decay*n)) for n in range(100)]
print(net_carbon)

years = np.array(range(100))+2014
global_carbon = [750+sum(net_carbon[:n]) for n in range(100)]
print(global_carbon)

temp = [math.log(g/750)/math.log(2) for g in global_carbon]
print(temp)

pop_rate = 1.007
us_pop = [318.9 *(pop_rate**n) for n in range(100)]
print(us_pop)

print(func(13,*popt)*us_pop[-1])

deaths_yearly = [(math.exp((13.3+temp[i])*popt[0]) - popt[1])*us_pop[i] for i in range(100)]

source = ColumnDataSource(data=dict(x=years, y=temp,txt=["79850"], kc_yr = deaths_yearly))

p2 = figure(tools='',width=700, height=300, title='Total Net Increase in Tempurature by Year (C)')
p2.line('x', 'y', source=source, color='red')

p2.toolbar_location = None

p3 = figure(tools='',width=700, height=130, title='GLOBAL WARMING KILL COUNT - 100 YEARS:')
p3.xgrid.grid_line_color = None
p3.ygrid.grid_line_color = None
p3.xaxis.axis_line_color = None
p3.yaxis.axis_line_color = None
p3.xaxis[0].formatter = bokeh.models.PrintfTickFormatter(format=" ")
p3.yaxis[0].formatter = bokeh.models.PrintfTickFormatter(format=" ")
p3.yaxis.minor_tick_line_color = None
p3.yaxis.major_tick_line_color = None
p3.xaxis.minor_tick_line_color = None
p3.xaxis.major_tick_line_color = None
p3.text(5,5,source=source,text='txt', text_font_size='50',x_offset=-55,y_offset=12)
p3.toolbar_location = None

p4  = figure(tools='',width=700, height=250, title='Year by Year Deaths Breakdown:')
p4.line('x', 'kc_yr', source=source, color='navy')
p4.toolbar_location = None

callback = bokeh.models.actions.Callback(args=dict(source=source), code="""
    var data = source.get('data');

    if (typeof window.carbon_increase == 'undefined')
    {
        window.carbon_increase = .029
    }
    if (typeof window.carbon_cutback == 'undefined')
    {
        window.carbon_cutback = 0.0005
    }


    if (typeof window.absorb_rate == 'undefined')
    {
        window.absorb_rate = .46
    }
    if (typeof window.absorb_decay == 'undefined')
    {
        window.absorb_decay = .01
    }

    x = data['x']
    y = data['y']
    txt = data['txt']
    kc_yr = data['kc_yr']
    regress = [ 0.05904493, 0.73107788]
    pop = [318.9, 321.13229999999993, 323.3802260999999, 325.6438876826999, 327.9233948964787, 330.21885866075405, 332.5303906713793, 334.8581034060789, 337.20211012992144, 339.5625249008308, 341.93946257513664, 344.3330388131626, 346.7433700848547, 349.1705736754486, 351.6147676911767, 354.07607106501496, 356.55460356247, 359.05048578740724, 361.56383918791903, 364.09478606223445, 366.64344956467005, 369.2099537116227, 371.794423387604, 374.3969843513172, 377.0177632417763, 379.65688758446873, 382.31448579756, 384.9906871981429, 387.6856220085299, 390.39942136258946, 393.1322173121276, 395.88414283331247, 398.6553318331456, 401.4459191559776, 404.2560405900694, 407.0858328741998, 409.9354337043192, 412.8049817402494, 415.6946166124311, 418.6044789287181, 421.534710281219, 424.48545325318753, 427.45685142595977, 430.44904938594146, 433.46219273164303, 436.4964280807645, 439.55190307732977, 442.62876639887105, 445.7271677636631, 448.8472579380087, 451.9891887435747, 455.1531130647797, 458.3391848562331, 461.5475591502267, 464.7783920642782, 468.03184080872813, 471.30806369438915, 474.60722014024986, 477.9294706812315, 481.2749769760001, 484.643901814832, 488.0364091275358, 491.4526639914285, 494.89283263936846, 498.357082467844, 501.8455820451188, 505.35850111943466, 508.8960106272706, 512.4582827016615, 516.045490680573, 519.6578091153369, 523.2954137791443, 526.9584816755983, 530.6471910473274, 534.3617213846586, 538.1022534343512, 541.8689692083916, 545.6620519928503, 549.4816863568001, 553.3280581612977, 557.2013545684267, 561.1017640504056, 565.0294763987584, 568.9846827335497, 572.9675755126844, 576.9783485412731, 581.017196981062, 585.0843173599294, 589.1799075814489, 593.304166934519, 597.4572961030605, 601.6394971757819, 605.8509736560122, 610.0919304716043, 614.3625739849056, 618.6631120027997, 622.9937537868193, 627.3547100633269, 631.7461930337702, 636.1684163850065]



    if (cb_obj.get('title')=="Carbon Increase Rate")
    {
        window.carbon_increase = cb_obj.get('value')
    }
    if (cb_obj.get('title')=="Carbon Percent Cutback Yearly")
    {
        window.carbon_cutback = cb_obj.get('value')
    }

    increases = []
    for (i = 0; i < x.length; i++)
    {
        mod = 1.0+Math.max(window.carbon_increase-window.carbon_cutback*i,-1.35)
        increases[i] = 30* Math.pow(mod,i) * Math.min(1, (window.absorb_rate+window.absorb_decay*i))
    }

    for (i = 0; i < x.length; i++)
    {
        sum = 0
        for (j = 0; j < i; j++)
        {
            sum+=increases[j]
        }
        y[i] = Math.log((750+sum)/750.0)/Math.log(2)
    }

    toll = 0

    for (k = 0; k < pop.length; k++)
    {
        kc_yr[k] = (Math.exp(regress[0]*(13.3+y[k]))-regress[1])*pop[k]
        toll += (Math.exp(regress[0]*(13.3+y[k]))-regress[1])*pop[k]
    }
    txt[0]=String(Math.round(toll))

    source.trigger('change');
""")

slider = Slider(start=0, end=.1, value=.029, step=.001,
                title="Carbon Increase Rate", callback=callback)

slider2 = Slider(start=0, end=.01, value=0.0005, step=.0001,
                title="Carbon Percent Cutback Yearly", callback=callback)

plt = bokeh.io.vform(p,slider,slider2,p2,p3,p4)


# show the results
show(plt)

from bokeh.embed import components
script, div = components(plt)

print()
print(script)
print()
print(div)



