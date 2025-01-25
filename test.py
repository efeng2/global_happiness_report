import import_data
import graphing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def test_load_in_data():
    """
    Tests the load in data method.
    """
    return import_data.load_in_data()


def test_visual_correlation(df_merged):
    """
    Tests the visual correlation method.
    """
    graphing.visual_correlation(df_merged)


def test_three_dimentional_correlation(df_merged):
    """
    Tests the three dimentional correlation method.
    """
    graphing.three_dimentional_correlation(df_merged)


def test_visual_ML(df_merged):
    """
    Tests the visual ML method.
    """
    graphing.visual_ML(df_merged)


def test_ML_liability(df_merged):
    """
    Prints the R-sqaured score of the models used in the machine learning
    models used to predict pre and post-pandemic data.
    """
    data_ml = df_merged.drop(["Happiness_Rank", "Country", "Region"], axis=1)
    data_ml = data_ml.astype(float)
    before_pandemic = data_ml[(data_ml['Year'] < 2020) &
                              (data_ml['Year'] >= 2015)]

    data_before_pandemic = before_pandemic.drop(["Year"], axis=1)
    data = data_before_pandemic.loc[:, data_before_pandemic.columns
                                    != 'Happiness_Score']
    happiness = data_before_pandemic["Happiness_Score"]

    data_train, data_test, happiness_train, happiness_test = \
        train_test_split(data, happiness, test_size=0.2)

    model = DecisionTreeRegressor()
    model.fit(data_train, happiness_train)
    print('Pre-Pandemic R^2 Score ' +
          str(model.score(data_test, happiness_test)))

    after_pandemic = data_ml[(data_ml['Year'] >= 2020) &
                             (data_ml['Year'] <= 2022)]
    data_after_pandemic = after_pandemic.drop(["Year"], axis=1)

    data = data_after_pandemic.loc[:, data_after_pandemic.columns
                                   != 'Happiness_Score']
    happiness = data_after_pandemic["Happiness_Score"]

    data_train, data_test, happiness_train, happiness_test = \
        train_test_split(data, happiness, test_size=0.2)

    model = DecisionTreeRegressor()
    model.fit(data_train, happiness_train)
    print('During Pandemic R^2 Score ' +
          str(model.score(data_test, happiness_test)))


def test_viz_coutry_trend(df_merged):
    """
    Tests the viz coutry trend method.
    """
    graphing.viz_coutry_trend(df_merged)


def main():
    df_merged = test_load_in_data()

    # Test if each function runs
    test_visual_correlation(df_merged)
    test_visual_ML(df_merged)
    test_three_dimentional_correlation(df_merged)
    test_viz_coutry_trend(df_merged)

    # Calculates R-squared score of regression model
    test_ML_liability(df_merged)


if __name__ == '__main__':
    main()
