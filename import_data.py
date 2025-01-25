'''
A file that conducts the necessary inputs of data needed for analysis.
'''

import pandas as pd


def load_in_data():
    '''
    Merges the data sets into a data frame and prunes it to make sure
    data is consistant and not nonexistant. Returns the merged data frame.
    '''
    df2015 = pd.read_csv('input/2015.csv')
    df2016 = pd.read_csv('input/2016.csv')
    df2017 = pd.read_csv('input/2017.csv')
    df2018 = pd.read_csv('input/2018.csv')
    df2019 = pd.read_csv('input/2019.csv')
    df2020 = pd.read_csv('input/2020.csv')
    df2021 = pd.read_csv('input/2021.csv')
    df2022 = pd.read_csv('input/2022.csv')

    yrs = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    dfs = [df2015, df2016, df2017, df2018, df2019, df2020, df2021, df2022]
    for i in range(0, 8):
        dfs[i]['Year'] = yrs[i]

    common_cols = ['Happiness_Rank', 'Country',
                   'Region', 'Happiness_Score',
                   'Economy_GDP_per_Capita',
                   'Family_Social_Support',
                   'Health_Life_Expectancy',
                   'Freedom', 'Trust_Government_Corruption',
                   'Generosity', 'Year']

    df2015 = df2015.rename(columns={'Family': 'Family_Social_Support',
                                    'Happiness Rank': 'Happiness_Rank',
                                    'Happiness Score': 'Happiness_Score',
                                    'Economy (GDP per Capita)':
                                    'Economy_GDP_per_Capita',
                                    'Health (Life Expectancy)':
                                    'Health_Life_Expectancy',
                                    'Trust (Government Corruption)':
                                    'Trust_Government_Corruption'})

    df2016 = df2016.rename(columns={'Family': 'Family_Social_Support',
                                    'Happiness Rank': 'Happiness_Rank',
                                    'Happiness Score': 'Happiness_Score',
                                    'Economy (GDP per Capita)':
                                    'Economy_GDP_per_Capita',
                                    'Health (Life Expectancy)':
                                    'Health_Life_Expectancy',
                                    'Trust (Government Corruption)':
                                    'Trust_Government_Corruption'})

    df2017 = df2017.rename(columns={'Happiness.Rank': 'Happiness_Rank',
                                    'Happiness.Score': 'Happiness_Score',
                                    'Economy..GDP.per.Capita.':
                                    'Economy_GDP_per_Capita',
                                    'Family': 'Family_Social_Support',
                                    'Health..Life.Expectancy.':
                                    'Health_Life_Expectancy',
                                    'Trust..Government.Corruption.':
                                    'Trust_Government_Corruption'})

    df2017 = df2017.merge(df2015[["Country", "Region"]], on="Country",
                          how="left")
    df2017["Region"] = df2017["Region"].fillna('-')

    df2018 = df2018.rename(columns={'Overall rank': 'Happiness_Rank',
                                    'Country or region': 'Country',
                                    'Score': 'Happiness_Score',
                                    'GDP per capita': 'Economy_GDP_per_Capita',
                                    'Social support': 'Family_Social_Support',
                                    'Healthy life expectancy':
                                    'Health_Life_Expectancy',
                                    'Freedom to make life choices': 'Freedom',
                                    'Perceptions of corruption':
                                    'Trust_Government_Corruption'})

    df2018 = df2018.merge(df2015[["Country", "Region"]], on="Country",
                          how="left")
    df2018["Region"] = df2018["Region"].fillna('-')

    df2019 = df2019.rename(columns={'Overall rank': 'Happiness_Rank',
                                    'Country or region': 'Country',
                                    'Score': 'Happiness_Score',
                                    'GDP per capita': 'Economy_GDP_per_Capita',
                                    'Social support': 'Family_Social_Support',
                                    'Healthy life expectancy':
                                    'Health_Life_Expectancy',
                                    'Freedom to make life choices': 'Freedom',
                                    'Perceptions of corruption':
                                    'Trust_Government_Corruption'})

    df2019 = df2019.merge(df2015[["Country", "Region"]], on="Country",
                          how="left")
    df2019["Region"] = df2019["Region"].fillna('-')

    df2020 = df2020.rename(columns={'Country name': 'Country',
                                    'Regional indicator': 'Region',
                                    'Ladder score': 'Happiness_Score',
                                    'Happiness Rank': 'Happiness_Rank',
                                    'Explained by: Social support':
                                    'Family_Social_Support',
                                    'Explained by: Healthy life expectancy':
                                    'Health_Life_Expectancy',
                                    'Explained by: Freedom to make life' +
                                    ' choices': 'Freedom',
                                    'Explained by: Perceptions of corruption':
                                    'Trust_Government_Corruption',
                                    'Explained by: Log GDP per capita':
                                    'Economy_GDP_per_Capita',
                                    'Explained by: Generosity': 'Generosity'})

    df2020['Happiness_Rank'] = [i for i in range(1, len(df2020.index)+1)]

    df2020 = df2020.loc[:, ~df2020.columns.duplicated(keep='last')]

    df2021 = df2021.rename(columns={'Country name': 'Country',
                                    'Regional indicator': 'Region',
                                    'Ladder score': 'Happiness_Score',
                                    'Explained by: Social support':
                                    'Family_Social_Support',
                                    'Explained by: Healthy life expectancy':
                                    'Health_Life_Expectancy',
                                    'Explained by: Freedom to make life' +
                                    ' choices': 'Freedom',
                                    'Explained by: Perceptions of corruption':
                                    'Trust_Government_Corruption',
                                    'Explained by: Log GDP per capita':
                                    'Economy_GDP_per_Capita',
                                    'Happiness Rank': 'Happiness_Rank',
                                    'Explained by: Generosity': 'Generosity'})

    df2021 = df2021.loc[:, ~df2021.columns.duplicated(keep='last')]
    df2021['Happiness_Rank'] = [i for i in range(1, len(df2021.index)+1)]

    df2022 = df2022.merge(df2015[["Country", "Region"]], on="Country",
                          how="left")
    df2022["Region"] = df2022["Region"].fillna('-')

    df2022 = df2022.rename(columns={'RANK': 'Happiness_Rank',
                                    'Happiness score':
                                    'Happiness_Score',
                                    'Explained by: GDP per capita':
                                    'Economy_GDP_per_Capita',
                                    'Explained by: Social support':
                                    'Family_Social_Support',
                                    'Explained by: Healthy life expectancy':
                                    'Health_Life_Expectancy',
                                    'Explained by: Freedom to make life' +
                                    ' choices': 'Freedom',
                                    'Explained by: Generosity': 'Generosity',
                                    'Explained by: Perceptions of corruption':
                                    'Trust_Government_Corruption'})

    df2022 = df2022.replace(',', '.', regex=True)

    dfs = [df2015[common_cols], df2016[common_cols], df2017[common_cols],
           df2018[common_cols], df2019[common_cols], df2020[common_cols],
           df2021[common_cols], df2022[common_cols]]

    df_merged = pd.concat([df2015[common_cols], df2016[common_cols],
                           df2017[common_cols], df2018[common_cols],
                           df2019[common_cols], df2020[common_cols],
                           df2021[common_cols], df2022[common_cols]])

    df_merged.dropna(axis='rows', inplace=True)
    df_merged.to_csv('world-happiness-report-2015-2022-cleaned.csv')
    return df_merged
