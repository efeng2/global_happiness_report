'''
A file that runs the full analysis on world happiness data and produces
several graphs in order to help analyze the happiness trends in each country.
'''

import import_data
import graphing


def main():
    df_merged = import_data.load_in_data()
    graphing.visual_correlation(df_merged)
    graphing.three_dimentional_correlation(df_merged)
    graphing.visual_ML(df_merged)
    graphing.viz_coutry_trend(df_merged)


if __name__ == '__main__':
    main()
