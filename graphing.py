'''
A file that takes care of the visualizations and analysis of world happiness
data.
'''

import pandas as pd
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
from sklearn.svm import SVR
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def visual_correlation(df_merged):
    '''
    Takes in the pruned dataframe and graphs the correlation between each
    factor and other factors, focusing on the relationship between each
    factor and its correlation with the happiness score.
    '''
    drop_rank = df_merged.drop(["Happiness_Rank", "Country", "Region", "Year"],
                               axis=1)

    drop_rank = drop_rank.astype(float)
    corr_matrix_happy = drop_rank.corr()
    trace_corr_happy = go.Heatmap(z=np.array(corr_matrix_happy),
                                  x=corr_matrix_happy.columns,
                                  y=corr_matrix_happy.columns,
                                  colorscale=[[0, 'rgb(190,247,255)'],
                                              [1, 'rgb(125, 166, 255)']])
    fig = go.Figure(trace_corr_happy)
    fig.update_layout(title='Correlation Between Different Factors and' +
                      ' Happiness Score')
    plot(fig, auto_open=False, filename='plots/correlation.html')


def three_dimentional_correlation(df_merged):
    '''
    Takes in the pruned dataframe and graphs the factors that contribute the
    most to the happiness score in a 3d graph.
    '''
    drop_rank = df_merged.drop(["Happiness_Rank", "Country", "Region", "Year"],
                               axis=1)

    drop_rank = drop_rank.astype(float)
    mesh_size = .02
    margin = 0

    X = drop_rank[['Economy_GDP_per_Capita', 'Health_Life_Expectancy']]
    y = drop_rank['Happiness_Score']

    model = SVR(C=1.)
    model.fit(X, y)

    x_min, x_max = (X.Economy_GDP_per_Capita.min() - margin,
                    X.Economy_GDP_per_Capita.max() + margin)
    y_min, y_max = (X.Health_Life_Expectancy.min() - margin,
                    X.Health_Life_Expectancy.max() + margin)
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    fig = px.scatter_3d(drop_rank,
                        x='Economy_GDP_per_Capita',
                        y='Health_Life_Expectancy',
                        z='Happiness_Score',
                        title='Relationship among Happiness Score, Economy,' +
                        'and Heath')
    fig.update_traces(marker=dict(size=5))
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
    plot(fig, auto_open=False, filename='plots/3d_plot.html')


def visual_ML(df_merged):
    '''
    Takes in the pruned dataframe and uses machine learning regression
    to find the trends between pre-pandamic data and during-pandamic
    data and graphs it.
    '''
    data_ml = df_merged.drop(["Happiness_Rank", "Country", "Region"], axis=1)
    data_ml = data_ml.astype(float)
    before_pandemic = data_ml[(data_ml['Year'] < 2020) &
                              (data_ml['Year'] >= 2015)]
    after_pandemic = data_ml[(data_ml['Year'] >= 2020) &
                             (data_ml['Year'] <= 2022)]
    data_before_pandemic = before_pandemic.drop(["Year"], axis=1)
    X = data_before_pandemic.drop("Happiness_Score", axis=1)
    lm_before = LinearRegression()
    lm_before.fit(X, data_before_pandemic.Happiness_Score)
    print("Estimated Intercept is", lm_before.intercept_)
    print("The number of coefficients in this model are", lm_before.coef_)
    coef_df = pd.DataFrame(list(zip(X.columns, lm_before.coef_)),
                           columns=['features', 'coefficients'])
    print(coef_df)

    data_after_pandemic = after_pandemic.drop(["Year"], axis=1)
    X = data_after_pandemic.drop("Happiness_Score", axis=1)
    lm_after = LinearRegression()
    lm_after.fit(X, data_after_pandemic.Happiness_Score)
    print("Estimated Intercept is", lm_after.intercept_)
    print("The number of coefficients in this model are", lm_after.coef_)
    coef_df = pd.DataFrame(list(zip(X.columns, lm_after.coef_)),
                           columns=['features', 'coefficients'])
    print(coef_df)

    data_year = data_ml.groupby('Year')['Happiness_Score'].mean().reset_index()
    lm_year = LinearRegression()
    data_year_before = data_year[data_year['Year'] <= 2019]
    data_year_after = data_year[data_year['Year'] > 2019]
    lm_year.fit(data_year_before[['Year']], data_year_before.Happiness_Score)
    print("Estimated Intercept is", lm_year.intercept_)
    print("The number of coefficients in this model are", lm_year.coef_)

    lm_pandemic = LinearRegression()
    lm_pandemic.fit(data_ml[['Year']], data_ml.Happiness_Score)
    print("Estimated Intercept is", lm_pandemic.intercept_)
    print("The number of coefficients in this model are", lm_pandemic.coef_)

    x = data_year.Year
    y = data_year.Happiness_Score

    x_before = data_year_before.Year
    y_before = data_year_before.Happiness_Score

    x_after = data_year_after.Year
    y_after = data_year_after.Happiness_Score

    a, b = np.polyfit(x, y, 1)
    a_before, b_before = np.polyfit(x_before, y_before, 1)
    a_after, b_after = np.polyfit(x_after, y_after, 1)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.scatter(x, y, s=10, c='lightblue', alpha=0.5)
    ax1.plot(x, a*x+b, c='b', alpha=0.5, label='2015-2022 fitted line')
    ax1.scatter(x_before, y_before, s=10, c='r', marker="o",
                label='before pandemic', alpha=0.5)
    ax1.plot(x_before, a_before*x_before+b_before, c='orange', alpha=0.5,
             label='before pandemic fitted line')
    ax1.scatter(x_after, y_after, s=10, c='g', marker="^",
                label='during pandemic', alpha=0.5)
    ax1.plot(x_after, a_after*x_after+b_after, c='g', alpha=0.5,
             label='during pandemic fitted line')
    plt.legend(loc='upper left')
    plt.xlabel('Year')
    plt.ylabel('Average Happiness Score in the World')
    plt.title('Trend of Average Happiness Score in the World in 2015 - 2022',
              loc='left')
    plt.text(2016.7, 5.34, 'y_before = ' + '{:.2f}'.format(b_before) +
             ' + {:.3f}'.format(a_before) + 'x_before', size=10)
    plt.text(2016.5, 5.445, 'y = ' + '{:.2f}'.format(b) +
             ' + {:.2f}'.format(a) + 'x', size=10)
    plt.text(2018.5, 5.55, 'y_covid = ' + '{:.2f}'.format(b_after) +
             ' + {:.2f}'.format(a_after) + 'x_covid', size=10)
    plt.savefig('plots/covid_trends_before_and_after.png')


def viz_coutry_trend(df_merged):
    '''
    Takes in the pruned dataframe and graphs the happiness scores of
    Malaysia and India over the years 2015 and 2022.
    '''
    df_malaysia = df_merged[df_merged['Country'] == 'Malaysia']
    df_india = df_merged[df_merged['Country'] == 'India']

    # plot
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot('Year', 'Happiness_Score', data=df_malaysia, linestyle='-',
             marker='o', alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel('Happiness Score')
    plt.xticks(rotation=-45)
    plt.title("Trend of Happiness Score of Malaysia in 2015-2022")

    plt.subplot(122)
    plt.plot('Year', 'Happiness_Score', data=df_india, linestyle='-',
             marker='o', color="orange", alpha=0.5)
    plt.title("Trend of Happiness Score of India in 2015-2022")
    plt.xlabel('Year')
    plt.ylabel('Happiness Score')
    plt.xticks(rotation=-45)
    plt.savefig('plots/trend_of_malyasia_and_india.png')
