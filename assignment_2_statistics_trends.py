#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Defining the dataset file path
file_path = 'data/API_19_DS2_en_csv_v2_5998250.csv'

# Defining the figure directory path, in which store the generated graphs pictures.
figure_dir = 'graphs'

# Defining countries
countries = {'BRA': 'Brazil', 'CAN': 'Canada'}

# Defining indicators, I will create separate DataFrames for each indicator
indicators = ['EN.URB.MCTY.TL.ZS', 'AG.LND.FRST.ZS']

# Define start year and end year
start_year = 2012
end_year = 2021


def load_and_transform_data(filepath):
    """
    This function first load the data from a dataset csv file using the pandas library.
    Extract the data, transposed the original dataframe and then  cleaned the transposed dataframe.
     After that it will convert the year column values to numeric.
    :param filepath:
    :return: df_original, df_transposed
    """
    # Loading the original DataFrame
    df_original = pd.read_csv(filepath, skiprows=4)
    # Transposing the DataFrame to get countries as columns
    index_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    df_transposed = df_original.set_index(index_cols).transpose()
    # Resetting the index to make the years a column instead of an index
    df_transposed.reset_index(inplace=True)
    # Renaming the 'index' column to 'Year'
    df_transposed.rename(columns={'index': 'Year'}, inplace=True)
    # Clean missing values from the transposed dataframe
    df_transposed = df_transposed.dropna()
    # Converting 'Year' to numeric
    df_transposed['Year'] = pd.to_numeric(df_transposed['Year'], errors='coerce')
    return df_original, df_transposed


def extract_specific_data(df_original, countries, indicators, start_year, end_year):
    """
    Extracts and returns data for specific countries, indicators, and years.
    :return: dataframe object of filtered data
    """
    # Filtering for the specified countries and indicators
    filtered_df = df_original[(df_original['Country Code'].isin(countries)) &
                              (df_original['Indicator Code'].isin(indicators))]
    # Selecting the range of years
    years = [str(year) for year in range(start_year, end_year + 1)]
    return filtered_df[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + years]


def urban_population_vs_forest_area():
    """
    This function first load and extract the data, transposed the original dataframe and then
    cleaned the transposed dataframe. After that it will extract the data for specific years.
    This function will show the urban population vs forest area for countries Brazil and Canada
    for the time period 2012 to 2021 by using line plot.
    :return: None
    """
    # Extract data from dataset
    df_original, df_transposed = load_and_transform_data(file_path)

    # Extracting the specific data
    specific_data = extract_specific_data(df_original, countries, indicators, start_year, end_year)
    print(specific_data)

    df_original, df_transposed = load_and_transform_data(file_path)
    specific_data = extract_specific_data(df_original, countries, indicators, start_year, end_year)

    # Convert year columns to numeric for plotting
    years = [str(year) for year in range(start_year, end_year + 1)]
    specific_data[years] = specific_data[years].apply(pd.to_numeric, errors='coerce')

    # Creating separate DataFrames for each indicator
    urban_population_data = specific_data[specific_data['Indicator Code'] == 'EN.URB.MCTY.TL.ZS']

    # describe the urban population dataframe
    print(urban_population_data.describe())

    forest_area_data = specific_data[specific_data['Indicator Code'] == 'AG.LND.FRST.ZS']

    # apply pandas describe method on the forest_area_data dataframe
    print(forest_area_data.describe())

    # Plotting
    # create new figure with and set size width is 12 & height is 6 inches.
    # The `dpi` parameter shows dots per inch or resolution for the figure
    plt.figure(figsize=(12, 6), dpi=144)

    # Plotting Urban Population
    for country in urban_population_data['Country Code'].unique():
        country_data = urban_population_data[urban_population_data['Country Code'] == country]
        plt.plot(years, country_data[years].values.flatten(),
                 marker='o', label=f'Urban Population {countries[country]}')

    # Plotting Forest Area
    for country in forest_area_data['Country Code'].unique():
        country_data = forest_area_data[forest_area_data['Country Code'] == country]
        plt.plot(years, country_data[years].values.flatten(),
                 marker='s', label=f'Forest Area {countries[country]}')

    # Adding titles and labels
    plt.title('Urban Population vs Forest Area ({}-{})'.format(start_year, end_year))
    # Set label for x-axis
    plt.xlabel('Year')
    # Set label for y-axis
    plt.ylabel('Percentage')
    # Rotate the x-axis labels
    plt.xticks(rotation=45)
    # Add plot legend
    plt.legend()
    # Enable the plot grid
    plt.grid(True)
    plt.tight_layout()
    # Save the plot image to `figures` directory/
    plt.savefig(f"{figure_dir}/Urban Population vs Forest Area")
    # Show the plot
    plt.show()


def average_urban_population_vs_average_forest_area():
    """
    This function first load and extract the data, transposed the original dataframe and then
    cleaned the transposed dataframe. This function will show the average urban population
    vs average forest area by using bar plots.
    :return: None
    """
    # Load and extract specific data
    df_original, df_transposed = load_and_transform_data(file_path)
    specific_data = extract_specific_data(df_original, countries, indicators, start_year, end_year)
    # Convert year columns to numeric
    years = [str(year) for year in range(int(start_year), int(end_year) + 1)]
    specific_data[years] = specific_data[years].apply(pd.to_numeric, errors='coerce')
    # Creating separate DataFrames for each indicator
    urban_population_data = specific_data[specific_data['Indicator Code'] == 'EN.URB.MCTY.TL.ZS']
    forest_area_data = specific_data[specific_data['Indicator Code'] == 'AG.LND.FRST.ZS']
    # Calculating average values for each country and indicator
    urban_population_avg = urban_population_data.groupby('Country Code')[years].mean().mean(axis=1)
    forest_area_avg = forest_area_data.groupby('Country Code')[years].mean().mean(axis=1)
    # Plotting Bar Charts
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 6), dpi=144)
    # Urban Population Bar Chart
    urban_population_avg.plot(kind='bar', ax=axes[0], color=['blue', 'green'])
    axes[0].set_title('Average Urban Population ({}-{})'.format(start_year, end_year))
    axes[0].set_ylabel('Percentage')
    # Forest Area Bar Chart
    forest_area_avg.plot(kind='bar', ax=axes[1], color=['red', 'orange'])
    axes[1].set_title('Average Forest Area ({}-{})'.format(start_year, end_year))
    axes[1].set_ylabel('Percentage')
    # General settings, rotate the x-axis labels
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Save the plot image to `figures` directory/
    plt.savefig(f"{figure_dir}/Average Urban Population vs Average Forest Area")
    # Show plot
    plt.show()


def show_stacked_area_plot_indicators():
    """
    This function first load and extract the data, transposed the original dataframe and then
    cleaned the transposed dataframe.  This function will generate stacked plots for showing
    indicators for countries Brazil & Canada, during the specific time period 2012 to 2021.
    :return: None
    """
    # Load and extract specific data
    df_original, df_transposed = load_and_transform_data(file_path)
    specific_data = extract_specific_data(df_original, countries, indicators, start_year, end_year)

    # Convert year columns to numeric for plotting
    years = [str(year) for year in range(start_year, end_year + 1)]
    specific_data[years] = specific_data[years].apply(pd.to_numeric, errors='coerce')

    # Creating separate DataFrames for each indicator
    urban_population_df = specific_data[specific_data['Indicator Code'] == 'EN.URB.MCTY.TL.ZS']
    forest_area_df = specific_data[specific_data['Indicator Code'] == 'AG.LND.FRST.ZS']

    # Preparing data for stacked area chart
    stacked_data = {}

    for country in countries:
        country_urban_data = urban_population_df[urban_population_df['Country Code'] == country][years].mean(axis=0)
        country_forest_data = forest_area_df[forest_area_df['Country Code'] == country][years].mean(axis=0)
        stacked_data[country] = pd.DataFrame(
            {'Urban Population': country_urban_data, 'Forest Area': country_forest_data})

    # Plotting Stacked Area Charts
    for country, data in stacked_data.items():
        # create new figure with and set size width is 10 & height is 6 inches.
        # The `dpi` parameter shows dots per inch or resolution for the figure
        plt.figure(figsize=(10, 6), dpi=144)
        plt.stackplot(data.index, data['Urban Population'], data['Forest Area'],
                      labels=['Urban Population', 'Forest Area'], alpha=0.8)
        # Define plot title
        title = f'Stacked Area Chart of Indicators for {countries[country]} ({start_year}-{end_year})'
        plt.title(title)
        # Set x-axis label
        plt.xlabel('Year')
        # Set y-axis label
        plt.ylabel('Percentage')
        # Plot legend and set its location to upper left
        plt.legend(loc='upper left')
        # Enable grid in plot
        plt.grid(True)
        plt.tight_layout()
        # Save the plot image to `figures` directory/
        plt.savefig(f"{figure_dir}/{title}")
        # Show plot
        plt.show()


def show_corr_urban_pop_vs_forest_area():
    """
    This function first load and extract the data, transposed the original dataframe and then
    cleaned the transposed dataframe. This function will show the correlation between urban
    population and forest area by making the heatmap.
    :return: None
    """

    # Load and extract specific data
    df_original, df_transposed = load_and_transform_data(file_path)
    specific_data = extract_specific_data(df_original, countries, indicators, start_year, end_year)

    # Convert year columns to numeric for plotting
    years = [str(year) for year in range(start_year, end_year + 1)]
    specific_data[years] = specific_data[years].apply(pd.to_numeric, errors='coerce')

    # Creating separate DataFrames for each indicator
    urban_population_data = specific_data[specific_data['Indicator Code'] == 'EN.URB.MCTY.TL.ZS']
    forest_area_data = specific_data[specific_data['Indicator Code'] == 'AG.LND.FRST.ZS']

    correlations = {
        'Country': [],
        'Urban Population': [],
        'Forest Area': [],

    }

    for country in countries:
        urban_avg = \
            urban_population_data[urban_population_data['Country Code'] == country][years].mean(axis=1).values[0]
        forest_avg = forest_area_data[forest_area_data['Country Code'] == country][years].mean(axis=1).values[0]
        correlations['Country'].append(country)
        correlations['Urban Population'].append(urban_avg)
        correlations['Forest Area'].append(forest_avg)

    correlation_data = pd.DataFrame(correlations)

    # Setting 'Country' as the index
    correlation_data.set_index('Country', inplace=True)

    # Computing the correlation
    correlation_matrix = correlation_data.corr()

    # Displaying the correlation matrix
    print(correlation_matrix)

    # Visualizing the correlation with a heatmap
    # Create new figure with and set size width is 7 & height is 6 inches.
    # The `dpi` parameter shows dots per inch or resolution for the figure
    plt.figure(figsize=(7, 6), dpi=144)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap between Urban Population and Forest Area')
    # Save the plot image to `figures` directory/
    plt.savefig(f"{figure_dir}/Correlation Heatmap between Urban Population and Forest Area")
    plt.show()


"""
__main__: This function is the entry point in python script.
"""
if __name__ == "__main__":
    # Call the function for showing Urban Population vs Forest Area
    urban_population_vs_forest_area()

    # Call a function for showing Average Urban Population vs Average Forest Area
    average_urban_population_vs_average_forest_area()

    # Call a function for showing Stacked Area Chart of Indicators for countries Brazil and Canada
    show_stacked_area_plot_indicators()

    # Call a function for showing Correlation Heatmap between Urban Population and Forest Area
    show_corr_urban_pop_vs_forest_area()
