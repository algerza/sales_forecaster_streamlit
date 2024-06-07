import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fuzzywuzzy import fuzz
from PIL import Image
from utils import *
from statistics import mode
from io import StringIO

base="white"

st.set_page_config(
    layout="wide",
    page_title="Time-series test",
    page_icon="📊",
) 

def run():
        """
        Sets up the dashboard and the side bar for all pages
        """

        ################################################################################
        ###                                                                          ###
        ###                         SET UP SIDEBAR                                   ###
        ###                                                                          ###
        ################################################################################

        with st.sidebar:   
            st.title("Time-series application")  
            markdown_text = """
            Testing. Developed by [Alvaro Ager](https://www.linkedin.com/in/alvaroager/)
            """
            st.markdown(markdown_text)



        # ################################################################################
        # ###                                                                          ###
        # ###                             INTRODUCTION                                 ###
        # ###                                                                          ###
        # ################################################################################
        
        # # Displat hero image
        # st.image('hero.jpg', use_column_width=True)

        # # Display markdown text
        # markdown_text = """
        # Inspired by the popular fitness app Strava, this Python application parallels its functionality within the financial domain. This application discerns analogous financial transactions over a span of time, affording users the ability to discern and visualize patterns within their spending habits. Developed by [Alvaro Ager](https://www.linkedin.com/in/alvaroager/)
        
        # The background comes from my experience analysing my own fitness acitivities (pace, routes, dates...), so I was wondering if similar functionality could be applied to my financial transactions to find patterns and insights. This is how the feature looks in Strava:
        # """
        # st.markdown(markdown_text)

        # # Display the image
        # st.image('https://support.strava.com/hc/article_attachments/4413210049933', use_column_width=True)
        
        # # Display markdown text
        # st.markdown("When you process your statement, the app will search for similar transactions, cluster them together, and allow you to select the group you want to visualise.")

        # # Display horizontal line
        # st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)



        ################################################################################
        ###                                                                          ###
        ###                             APPLICATION                                  ###
        ###                                                                          ###
        ################################################################################

        # Streamlit app
        st.title("Load your data and play around!")
        st.subheader("Select a pre-loaded dataset or upload a file")


        # Read the data into a pandas DataFrame
        data = pd.read_csv('time_series_data_prod.csv', dayfirst=True)

        # List of available DataFrames
        dataframes = {'DataFrame 1': data}

        ################################################################################
        ###                                                                          ###
        ###               CREATE FILTER TO SELECT DATAFRAME OR CSV UPLOAD            ###
        ###                                                                          ###
        ################################################################################

        # Option to select a pre-loaded DataFrame
        selected_option = st.radio("Select Option:", ["Select Pre-loaded DataFrame", "Upload CSV File"])
 
        df = None

        if selected_option == "Select Pre-loaded DataFrame":
            # Allow the user to select a DataFrame
            df = data

        elif selected_option == "Upload CSV File":

        ################################################################################
        ###                                                                          ###
        ###                       INSTRUCTIONS  IF CSV FILE                          ###
        ###                                                                          ###
        ################################################################################
            # File uploader for CSV files
            # Display title 
            st.title('Instructions')

            # Upload CSV file
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

            if uploaded_file is not None:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(uploaded_file)


        # Display the selected data
        if df is not None:
                st.write("This is your input data:")
                st.write(df)

                # Create columns for date selection and target value selection
                date_col, target_col = st.columns(2)

                # Select date field in the first column
                with date_col:
                    date_column = st.selectbox("Select the Date Field:", options=df.columns)

                # Select target value field in the second column
                with target_col:
                    target_value_column = st.selectbox("Select the Target Value Field:", options=df.columns)

                # Multi-select box to choose exogenous variables
                exogenous_columns = st.multiselect("Select Exogenous Variables:", options=df.columns)

                # Options for prediction parameters
                st.header("Prediction Parameters")
                periods_to_predict = st.number_input("Periods to Predict:", min_value=1, value=30, step=1)
                end_training_date = st.date_input("End Training Date:", pd.to_datetime('2023-03-19'))

                # Convert 'date_column' to datetime type
                df[date_column] = pd.to_datetime(df[date_column])




                # Filter data before the determined end training date
                data_train = df[df[date_column] < pd.to_datetime(end_training_date)]

                # Create date with end_training_date + periods_to_predict
                last_forecasting_date = data_train[date_column].max() + pd.Timedelta(days=periods_to_predict + 1)

                # Create dataframe with all data (past and future). This is important because we need to feed the 
                #   models with future regressors
                all_data = df[df[date_column] < last_forecasting_date]

                # Create dataframe with all available data until end training date
                data_test = df[(df[date_column] < last_forecasting_date) & (df[date_column] >= pd.to_datetime(end_training_date))]






                # Visualize filtered data based on selected fields and exogenous variables
                if st.button("Predict"):
                    # Filter the data DataFrame based on the selected columns
                    selected_columns = [date_column, target_value_column] + exogenous_columns
                    filtered_data = df[selected_columns]


                    # Filter data before the determined end training date
                    data_train = filtered_data[filtered_data[date_column] < pd.to_datetime(end_training_date)]

                    # Create date with end_training_date + periods_to_predict
                    last_forecasting_date = data_train[date_column].max() + pd.Timedelta(days=periods_to_predict + 1)

                    # Create dataframe with all data (past and future). This is important because we need to feed the 
                    #   models with future regressors
                    all_data = filtered_data[filtered_data[date_column] < last_forecasting_date]

                    # Create dataframe with all available data until end training date
                    data_test = filtered_data[(filtered_data[date_column] < last_forecasting_date) & (filtered_data[date_column] >= pd.to_datetime(end_training_date))]


                    # Display the filtered DataFrame using st.table()
                    # st.write(f"Filtered Data based on '{date_column}', '{target_value_column}', and exogenous variables:")
                    # st.table(data_test.tail(5))  # Display the first 5 rows of the filtered DataFrame

                    # st.table(data[exogenous_columns])

                    ################################################################################
                    ###                                                                          ###
                    ###                       SKFORECAST                                         ###
                    ###                                                                          ###
                    ################################################################################


                    data_train = data_train.interpolate(method='ffill')

                    def process_data_skforecast(df, date_col):
                        """
                        Process the DataFrame by converting the specified date column to datetime format,
                        setting it as the index, resampling to daily frequency, and sorting by date.
                        
                        Args:
                        df (DataFrame): Input DataFrame with a date column
                        date_col (str): Name of the date column
                        
                        Returns:
                        DataFrame: Processed DataFrame with datetime index
                        """
                        # Convert specified date column to datetime format
                        st.write(f"Processing with date column: {date_col}")
                        st.write(f"Columns in DataFrame: {df.columns.tolist()}")
                        
                        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
                        
                        # Set specified date column as index
                        df = df.set_index(date_col)
                        
                        # Resample to daily frequency
                        df = df.asfreq('D')
                        
                        # Sort DataFrame by date
                        df = df.sort_index()
                        
                        return df

                    # Apply function over datasets for skforecast
                    data_train_skforecast = process_data_skforecast(data_train, date_column)
                    data_test_skforecast = process_data_skforecast(data_test, date_column)

                    # Import libraries
                    from skforecast.ForecasterAutoreg import ForecasterAutoreg
                    from sklearn.ensemble import RandomForestRegressor

                    # Create and fit forecaster
                    # ==============================================================================
                    forecaster = ForecasterAutoreg(
                                    regressor = RandomForestRegressor(random_state=123),
                                    lags      = 14
                                )
                    
                    # # selected_columns = [date_column, target_value_column] + exogenous_columns
                    # # filtered_data = data[selected_columns]

                    forecaster.fit(
                        y    = data_train_skforecast[target_value_column],
                        exog = data_train_skforecast[exogenous_columns]
                    )

                    # Predict
                    # ==============================================================================
                    steps = periods_to_predict
                    predictions = forecaster.predict(
                                    steps = steps,
                                    exog = data_test_skforecast[exogenous_columns]
                                )
                    
                    # st.text(predictions)

                    # Create a figure using Plotly graph objects
                    fig = go.Figure()

                    # Add the forecasted values as a trace on the figure
                    fig.add_trace(go.Scatter(x=data_test_skforecast.index, y=predictions,
                                            mode='lines', name='Forecast', line=dict(dash='dash')))

                    # Update figure layout
                    fig.update_layout(
                        xaxis_title='Date',
                        yaxis_title='Units Sold',
                        title='Actual vs Forecasted Values',
                        xaxis_tickangle=45  # Rotate x-axis labels for better readability
                    )

                    # Display the plot using Streamlit
                    st.plotly_chart(fig, use_container_width=True)



                    importance_table = forecaster.get_feature_importances().sort_values(by='importance', ascending=False)

                    st.table(importance_table)














        # # Display the selected data
        # if 'df' in locals():
        #     st.write("This is your input data:")
        #     st.write(df)

        #     # Dropdown menu to select the date field
        #     date_column = st.selectbox("Select the Date Field:", options=df.columns)

        #     # Continue with your analysis or visualization using the selected date_column
        #     if st.button("Visualize Data"):
        #         # Example: Plot data using Plotly based on the selected date_column
        #         # (Replace this with your actual visualization code)
        #         date_field = {date_column}
        #         st.table(data[date_field].head(2))


        #     # Dropdown menu to select the date field
        #     target_value_column = st.selectbox("Select the Date ield:", options=df.columns)

        #     # Continue with your analysis or visualization using the selected date_column
        #     if st.button("Visualiz Data"):
        #         # Example: Plot data using Plotly based on the selected date_column
        #         # (Replace this with your actual visualization code)
        #         target_value_field = {target_value_column}
        #         st.table(data[target_value_field].head(2))
            

        #     st.table(data[[{date_column}, {target_value_column}]])



























        # ################################################################################
        # ###                                                                          ###
        # ###                     APPLY FUNCTIONS AND MATCH THE ROWS                   ###
        # ###                                                                          ###
        # ################################################################################

        #     # Convert the "Started Date" column to datetime
        #     df["Started Date"] = pd.to_datetime(df["Started Date"])
        #     df["Amount"] = round(df["Amount"].astype('int64'), 2)

        #     # Data preprocessing
        #     add_day_of_week(df)
        #     calculate_perc_columns(df)

        #     df['day_of_week'] = df['day_of_week'].map(day_mapping)

        #     # Find matching rows and create 'matched_rows' column
        #     df['matched_rows'] = df.apply(lambda row: find_matching_rows(row, df), axis=1)
        #     df['matched_rows'] = df['matched_rows'].apply(lambda x: sorted(x) if isinstance(x, list) else [])

        #     required_columns = ['Amount', 'Started Date', 'Description']



        # ################################################################################
        # ###                                                                          ###
        # ###               CREATE LOGIC FOR THE MERCHANT AND ROW FILTERS              ###
        # ###                                                                          ###
        # ################################################################################

        #     if all(col in df.columns for col in required_columns) and df.shape[0] > 0:

        #         # Filter merchants with non-empty 'matched_rows'
        #         filtered_merchants = df[df['matched_rows'].apply(lambda x: len(x) > 0)]['Description'].dropna().unique()
        #         if len(filtered_merchants) == 0:
        #             st.warning("No merchants with matching rows found.")
        #             return

        #         selected_merchant = st.selectbox("Select a merchant / person:", filtered_merchants)

        #         # Filter rows based on selected merchant
        #         pre_filtered_df = df[df['Description'] == selected_merchant]

        #         # Filter rows with non-empty 'matched_rows'
        #         pre_filtered_df = pre_filtered_df = pre_filtered_df[pre_filtered_df['matched_rows'].apply(lambda x: isinstance(x, list) and len(x) >= 0)].reset_index()

        #         # pre_filtered_df = pre_filtered_df.reset_index()

        #         if pre_filtered_df.shape[0] == 0:
        #             st.warning(f"No matching rows found for {selected_merchant}.")
        #             return

        #         # Create a selection dropdown for rows
        #         selected_row = st.selectbox("Select a transaction:", pre_filtered_df[pre_filtered_df['matched_rows'].apply(lambda x: len(x) > 0)].dropna()['index'].unique())

        #         selected_row_indices = pre_filtered_df[pre_filtered_df['index'] == selected_row].matched_rows.iloc[0]

        #         filtered_df = pre_filtered_df[pre_filtered_df['index'].isin(selected_row_indices)]



        # ################################################################################
        # ###                                                                          ###
        # ###               CALCULATE INSIGHTS ON THE FLY BASED ON FILTERS             ###
        # ###                                                                          ###
        # ################################################################################

        #         # Find matching names and create 'matched_names' column
        #         filtered_df['matched_names'] = filtered_df.apply(lambda row: find_matching_column(row, filtered_df, 'Description'), axis=1)
        #         filtered_df['mode_matched_names'] = filtered_df['matched_names'].apply(lambda x: mode(x) if len(x) > 0 else None)

        #         # Find matching days of the week and create 'matched_days_week' column
        #         filtered_df['matched_days_week'] = filtered_df.apply(lambda row: find_matching_column(row, filtered_df, 'week_category'), axis=1)
        #         filtered_df['mode_matched_days_week'] = filtered_df['matched_days_week'].apply(lambda x: mode(x) if len(x) > 0 else None)


        #         # Calculate time differences between consecutive dates
        #         filtered_df["Time Difference"] = filtered_df["Started Date"].diff()

        #         # Calculate the average time difference
        #         average_time_diff = filtered_df["Time Difference"].mean()

        #         # Display horizontal line
        #         st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

        #         # Display section with KPIs
        #         col1, col2, col3, col4 = st.container().columns(4)
        #         with col1:
        #             st.metric(label = "Transactions in this group", value = filtered_df['Amount'].count())
        #         with col2:
        #             st.metric(label = "Average amount", value = round(filtered_df['Amount'].mean(), 2))
        #         with col3:
        #             st.metric(label = "When transaction usually happen", value = filtered_df['mode_matched_days_week'].iloc[0])
        #         with col4:
        #             st.metric(label = "Avg days between transactions", value = average_time_diff.days)
                
        #         # Display horizontal line   
        #         st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)



        # ################################################################################
        # ###                                                                          ###
        # ###                      VISUALISATION: CHART AND TABLE                      ###
        # ###                                                                          ###
        # ################################################################################

        #         # Scatter plot
        #         fig = go.Figure()

        #         # Scatter plot for 'Amount'
        #         fig.add_trace(go.Scatter(
        #             x=filtered_df['Started Date'],
        #             y=filtered_df['Amount'],
        #             mode='markers',
        #             text=filtered_df['Description'],
        #             marker=dict(
        #                 size=10,
        #                 color='darkorange',
        #                 opacity=0.7
        #             ),
        #             name='Amount'
        #         ))

        #         # Line trace for the average in orange
        #         fig.add_trace(go.Scatter(
        #             x=filtered_df['Started Date'],
        #             y=[filtered_df['Amount'].mean()] * len(filtered_df),  # Same average value for all points
        #             mode='lines',
        #             line=dict(color='lightblue', width=2),
        #             name='Average'
        #         ))

        #         fig.update_layout(
        #             xaxis=dict(title='Date'),
        #             yaxis=dict(title='Amount')
        #         )

        #         # Display the plot in Streamlit
        #         st.plotly_chart(fig, use_container_width=True)

        #         # Display the filtered DataFrame
        #         st.table(filtered_df[['Description', 'Started Date', 'Amount', 'day_of_week']].sort_values(by='Started Date'))

        #     # Show warning messages if the input dataframe is incorrect
        #     else:
        #         missing_columns = [col for col in required_columns if col not in df.columns]
        #         if df.shape[0] == 0:
        #             st.warning("The DataFrame has no rows of data (excluding headers).")
        #         elif missing_columns:
        #             st.warning(f"The following columns are missing: {', '.join(missing_columns)}")

run()   