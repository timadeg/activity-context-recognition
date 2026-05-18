#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1. STATISTICAL FEATURES USING USER-DEFINED FUNCTIONS

# In[8]:


class DescriptiveStatistics:
    """
    This class provides methods for calculating descriptive statistics of a given dataset.
    """
    def __init__(self, data):
        self.data = data

        
    def mean(self, data_column):
        """
        Calculates the mean (average) of a given data column.

        Args:
            data_column: The column of data for which the mean is to be calculated.

        Returns:
            The mean value of the data column.
        """
        try:
            return sum(data_column) / len(data_column)
        except Exception as e:
            print("An exception occurred while calculating mean: ", str(e))

    def median(self, data_column):
        """
        Calculates the median of a given data column.

        Args:
            data_column: The column of data for which the median is to be calculated.

        Returns:
            The median value of the data column.
        """
        try:
            sorted_data = sorted(data_column)
            length = len(sorted_data)
            if length % 2 == 0:
                return (sorted_data[length // 2 - 1] + sorted_data[length // 2]) / 2
            else:
                return sorted_data[length // 2]
        except Exception as e:
            print("An exception occurred while calculating median: ", str(e))
            

    def variance(self, data_column):
        """
        Calculates the variance of a given data column.

        Args:
            data_column: The column of data for which the variance is to be calculated.

        Returns:
            The variance value of the data column.
        """
        try:
            mean_val = self.mean(data_column)
            return sum((xi - mean_val) ** 2 for xi in data_column) / len(data_column)
        except Exception as e:
            print("An exception occurred while calculating variance: ", str(e))
            

    def std_dev(self, data_column):
        """
        Calculates the standard deviation of a given data column.

        Args:
            data_column: The column of data for which the standard deviation is to be calculated.

        Returns:
            The standard deviation value of the data column.
        """
        try:
            return self.variance(data_column) ** 0.5
        except Exception as e:
            print("An exception occurred while calculating standard deviation: ", str(e))
            

    def rms(self, data_column):
        """
        Calculates the root mean square of a given data column.

        Args:
            data_column: The column of data for which the root mean square is to be calculated.

        Returns:
            The root mean square value of the data column.
        """
        try:
            return (sum(x**2 for x in data_column) / len(data_column)) ** 0.5
        except Exception as e:
            print("An exception occurred while calculating root mean square: ", str(e))

    def cv(self, data_column):
        """
        Calculates the coefficient of variation of a given data column.

        Args:
            data_column: The column of data for which the coefficient of variation is to be calculated.

        Returns:
            The coefficient of variation value of the data column.
        """
        try:
            mean_val = self.mean(data_column)
            std_dev_val = self.std_dev(data_column)
            return std_dev_val / mean_val
        except Exception as e:
            print("An exception occurred while calculating coefficient of variation: ", str(e))

    def iqr(self, data_column):
        """
        Calculates the interquartile range of a given data column.

        Args:
            data_column: The column of data for which the interquartile range is to be calculated.

        Returns:
            The interquartile range value of the data column.
        """
        try:
            sorted_data = sorted(data_column)
            length = len(sorted_data)
            Q1 = sorted_data[int(length * 0.25)]
            Q3 = sorted_data[int(length * 0.75)]
            return Q3 - Q1
        except Exception as e:
            print("An exception occurred while calculating interquartile range: ", str(e))

    def calculate_covariance(self, col1, col2):
        """
        Calculates the covariance between two columns of the dataset.

        Args:
            col1: The name of the first column.
            col2: The name of the second column.

        Returns:
            The covariance value between col1 and col2.
        """
        try:
            mean_col1 = self.mean(self.data[col1])
            mean_col2 = self.mean(self.data[col2])
            covariance = sum((self.data[col1][i] - mean_col1) * (self.data[col2][i] - mean_col2) for i in range(len(self.data))) / (len(self.data) - 1)
            return covariance
        except Exception as e:
            print("An exception occurred while calculating covariance: ", str(e))

    def min_val(self, data_column):
        """
        Finds the minimum value in a given data column.

        Args:
            data_column: The column of data in which to find the minimum value.

        Returns:
            The minimum value in the data column.
        """
        try:
            min_value = min(data_column)
            return min_value
        except Exception as e:
            print("An exception occurred while finding the minimum value: ", str(e))

    def max_val(self, data_column):
        """
        Finds the maximum value in a given data column.

        Args:
            data_column: The column of data in which to find the maximum value.

        Returns:
            The maximum value in the data column.
        """
        try:
            max_value = max(data_column)
            return max_value
        except Exception as e:
            print("An exception occurred while finding the maximum value: ", str(e))

    def print_statistics(self, column):
        """
        Prints various statistics of a given column.

        Args:
            column: The name of the column for which to print the statistics.

        Raises:
            ValueError: If the column is not found in the dataset.
        """
        try:
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame")
            
            column_data = self.data[column]
            
            print(f"Mean of {column}: {self.mean(column_data)}")
            print(f"Median of {column}: {self.median(column_data)}")
            print(f"Variance of {column}: {self.variance(column_data)}")
            print(f"Standard Deviation of {column}: {self.std_dev(column_data)}")
            print(f"Root Mean Square of {column}: {self.rms(column_data)}")
            print(f"Interquartile Range of {column}: {self.iqr(column_data)}")
            print(f"Coefficient of Variation of {column}: {self.cv(column_data)}")
            print(f"Minimum of {column}: {self.min_val(column_data)}")
            print(f"Maximum of {column}: {self.max_val(column_data)}")
        except Exception as e:
            print("An exception occurred while printing statistics: ", str(e))

    def print_covariance(self, col1, col2):
        """
        Prints the covariance between two columns.

        Args:
            col1: The name of the first column.
            col2: The name of the second column.
        """
        print(f"Covariance between {col1} and {col2}: {self.calculate_covariance(col1, col2)}")


# ## 2. EXPLORATORY DATA ANALYSIS

# In[10]:


class DataVisualization:
    """
    Class for visualizing various aspects of a dataset.
    """
    def __init__(self, data):
        """
        Initialize the class with a pandas DataFrame.

        Args:
            data: pandas DataFrame to be visualized.
        """
        self.data = data

    def activity_distribution(self, column='activity'):
        """
        Plot the distribution of values in the given column.

        Args:
            column: String, name of the column to plot. Default is 'activity'.
        """
        try:
            # Count the occurrences of each unique value in the column
            activity_counts = self.data[column].value_counts()
            
            # Prepare a plot with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))

            # Plot the bar chart of the activity counts in the first subplot
            activity_counts.plot(kind='bar', ax=axes[0])
            axes[0].set_xlabel('Activities')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Distribution of {} Column'.format(column))

            # Plot the bar chart of the activity counts (log scale) in the second subplot
            activity_counts.plot(kind='bar', ax=axes[1])
            axes[1].set_xlabel('Activities')
            axes[1].set_ylabel('Frequency (log scale)')
            axes[1].set_title('Distribution of {} Column (log scale)'.format(column))
            axes[1].set_yscale('log')  

            # Show the plot
            plt.show()

        except Exception as e:
            print("An exception occurred while plotting: ", str(e))

    def skew_kurt_analysis(self):
        """
        Perform skewness and kurtosis analysis on the numeric columns of the DataFrame.
        """
        # Calculate skewness and kurtosis
        skewness = self.data.skew(numeric_only=True)
        kurtosis = self.data.kurt(numeric_only=True)
        
        # Display the skewness and kurtosis values
        df_skew_kurt = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis})
        print("\n", df_skew_kurt)

        # Plot skewness and kurtosis
        fig, axes = plt.subplots(1, 2, figsize=(16, 3))
        sns.barplot(x=skewness.index, y=skewness.values, ax=axes[0])
        axes[0].set_title("Skewness")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
        axes[0].set_ylabel('Skewness value')

        sns.barplot(x=kurtosis.index, y=kurtosis.values, ax=axes[1])
        axes[1].set_title("Kurtosis")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
        axes[1].set_ylabel('Kurtosis value')

        # Arrange the plots properly and show them
        plt.tight_layout()
        plt.show()


        
    def activity_sensor_plot(self, sensor_columns, activity_column='activity'):
        """
        Plot the mean of sensor columns grouped by the activity column.

        Args:
            sensor_columns: List of column names of the sensors.
            activity_column: The column name of activities.
        """
        try:
            # Check if the columns are present in the DataFrame
            for column in sensor_columns:
                if column not in self.data.columns:
                    raise ValueError(f"Column '{column}' not found in the DataFrame")
            
            # Check if the activity_column is present in the DataFrame
            if activity_column not in self.data.columns:
                raise ValueError(f"Column '{activity_column}' not found in the DataFrame")
            
            # Compute the mean of sensor_columns grouped by the activity_column
            grouped_means = self.data.groupby(activity_column)[sensor_columns].mean()

            # Plot grouped_means as subplots
            grouped_means.plot(kind='bar', figsize=(14, 8), subplots=True, layout=(3, 2), legend=False)
            plt.suptitle("Grouped Bar Plots of Activities vs Sensor Variables")
            plt.show()
        except Exception as e:
            print("An exception occurred while running activity_sensor_plot: ", str(e))

    def plot_histograms(self):
        """
        Plot histograms of all columns of the DataFrame.
        """
        try:
            # Check if the input data is a pandas DataFrame
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError("Input should be a pandas DataFrame")

            # Create a figure to hold the subplots
            plt.figure(figsize=(15, 8))

            # Loop over all columns (except the last one) and plot a histogram for each
            for i, column in enumerate(self.data.columns[:-1]):
                plt.subplot(5, 4, i+1)
                sns.histplot(data=self.data, x=column, kde=True)
                plt.title(f'{column}')

            # Layout the plots nicely and display them
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print("An exception occurred while creating histograms: ", str(e))

    def plot_correlation_matrix(self):
        """
        Plot a correlation matrix of the DataFrame.
        """
        try:
            # Check if the input data is a pandas DataFrame
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError("Input should be a pandas DataFrame")

            # Create a figure to hold the plot
            plt.figure(figsize=(12, 6))

            # Plot a heatmap of the correlation matrix
            sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap='BuPu', vmin=-1, vmax=1)

            # Add a title and display the plot
            plt.title("Correlation Matrix")
            plt.show()

        except Exception as e:
            print("An exception occurred while creating the correlation matrix plot: ", str(e))


    def plot_box_plots(self):
        """
        Plots box plots for all columns in the DataFrame.

        Box plots are used to show overall patterns of response for a group. They provide a useful way to visualise the range and other characteristics of responses for a large group.
        """
        try:
            # Check if the input data is a pandas DataFrame. If not, raise an error.
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError("Input should be a pandas DataFrame")

            # Create a figure to hold the subplots.
            plt.figure(figsize=(15, 8))

            # Loop over all columns (except the last one) and plot a box plot for each.
            # Box plots will be arranged in a grid that is 5 plots tall and 4 plots wide.
            for i, column in enumerate(self.data.columns[:-1]):
                plt.subplot(5, 4, i+1)
                sns.boxplot(data=self.data, x=column)
                plt.title(f'{column}')  # Set the title of each subplot to the column name.

            # Adjust the subplots to fit in the figure area.
            plt.tight_layout()
            
            # Display the figure with all box plots.
            plt.show()
            
        except Exception as e:
            # Print the exception message if an exception occurred.
            print("An exception occurred while creating the box plots: ", str(e))


# In[ ]:




