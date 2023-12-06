import streamlit as st
import pandas as pd
from src.data_management import load_telco_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
    predict_binary_output, predict_profits)


def page_prospect_body3():


    # load predict PL files
    version = 'v3'
    profit_pip = load_pkl_file(
        f"outputs/predict_profits/{version}/clf_pipeline.pkl")
    profit_labels_map = load_pkl_file(
        f"outputs/predict_profits/{version}/label_map.pkl")
    profit_features = (pd.read_csv(f"outputs/predict_profits/{version}/X_train.csv")
                       .columns
                       .to_list()
                       )

   
    st.write("### Trade Analysis Interface")
    st.info(
        f"* Trader is interested if current price is a buy or sell position yeilding profit in a 4hr timeframe.\n"
        f"* Website will provide probability how likely it is a buy or sell position.\n"
        f"* The data used in the model is transformed to a trade strategy based on observing MA20, MA50, MA100 and bollinger band."
    )
    st.write("---")
    # Generate Live Data
    # check_variables_for_UI(profit_features, trade_features, cluster_features)
    X_live = DrawInputsWidgets()
    print(type(X_live))


      
        # predict on live data
    if st.button("Run Predictive Analysis with Direct Prices"):
        
        predict_profits(X_live, profit_features,
                                profit_pip, profit_labels_map)
    

def check_variables_for_UI(profit_features, trade_features, cluster_features):
    import itertools

    # The widgets inputs are the features used in all pipelines (tenure, churn, cluster)
    # We combine them only with unique values
    combined_features = set(
        list(
            itertools.chain(profit_features, trade_features, cluster_features)
        )
    )
    st.write(
        f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")


def DrawInputsWidgets():

    # Streamlit app
    st.title("Model Parameter Input")

    # load dataset
    df = load_telco_data()
   # print(df['close_ma50_1_diff_1'])
    percentageMin, percentageMax = 0.4, 2.0

# we create input widgets only for 6 features
    col1, col2, col3, col4 = st.beta_columns(4)
    col5, col6, col7, col8 = st.beta_columns(4)
    col9, col10, col11, col12 = st.beta_columns(4)

    # We are using these features to feed the ML pipeline - values copied from check_variables_for_UI() result

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type (numerical or categorical)
    # and set initial values
    with col1:
        feature = 'open_close_diff_1'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[0],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1
        )
    X_live[feature] = st_widget

    with col2:
        feature = 'open_close_diff1_lag1'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[1],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1
        )
    X_live[feature] = st_widget
    
    with col3:
        feature = 'close_ma50_1_diff_1'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[2],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1
        )
    X_live[feature] = st_widget


    with col4:
        feature = 'bb_status_1'
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            # index=3  # Set a default index if needed
            )
    X_live[feature] = st_widget

    with col5:
        feature = 'up_bb20_low_bb20_diff_1'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[4],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1
        )
    X_live[feature] = st_widget
    
    with col6:
        feature = 'trend_strength_1'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[5],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=1
        )
    X_live[feature] = st_widget
    
    with col7:
        feature = 'lagged_close_1'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[6],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1
        )
    X_live[feature] = st_widget
    
    with col8:
        feature = 'hr'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[7],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=1
        )
    X_live[feature] = st_widget
    
    with col9:
        feature = 'up_bb20_low_bb20_diff_4'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[8],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1
        )
    X_live[feature] = st_widget
    
    with col10:
        feature = 'ma50_4_ma100_4_diff_4'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[9],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1
        )
    X_live[feature] = st_widget
    
    with col11:
        feature = 'ma20_4_ma50_4_diff_4'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[10],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1
        )
    X_live[feature] = st_widget
    
    with col12:
        feature = 'close_ma100_4_diff_4'
        st_widget = st.number_input(
            label=feature,
            value=df[feature].iloc[11],  # Set a default value if needed
            min_value=df[feature].min(),
            max_value=df[feature].max(),
            step=0.1
        )
    X_live[feature] = st_widget

    # st.write(X_live)

    return X_live


def bb_status_labelling(close, high, low, upper_bb, lower_bb):
    
    if (high > upper_bb) & (low < upper_bb):
        bb_status = 'upper_hit'
    elif (high > lower_bb) & (low < lower_bb):
        bb_status = 'lower_hit'
    elif (upper_bb - high) < 0.01:
        bb_status = 'lower_near'
    elif (lower_bb - low) < 0.01:
        bb_status = 'lower_near'
    elif (close > upper_bb):
        bb_status = 'upper_crossover'
    elif (close < lower_bb):
        bb_status = 'lower_crossover'
    else:
        bb_status = 'inside_bb'
        
    return bb_status


def market_state_labelling(prev_close, close):
    slope = close - prev_close
    
    if slope > 0.05:
        market_state = "bullish"
    elif slope <-0.05:
        market_state = "bearish"
    else:
        market_state = "flat"
    
    return market_state


def user_input_processing():

    # Streamlit app
    st.title("Price Input")

    # Create an empty DataFrame to store user input
    price_input_df = pd.DataFrame(columns=["Type", "Value"])

    # Price types
    price_types = ["Hour", "Previous hr closed price", "Current hr closed price", "high price", "low price", "MA20", "MA50", "MA100", "Upper BB", "Lower BB"]

    # Allow the user to input values for each price type
    for price_type in price_types:
        value = st.number_input(f"{price_type}:", min_value=0.0, step=0.1)
        
        # Add the entered values to the DataFrame
        price_input_df = pd.concat([price_input_df, pd.DataFrame({"Type": [price_type], "Value": [value]})], ignore_index=True)

    # Display the entered values
    st.write("You entered:")
    st.write(price_input_df)
    
    # Extract values for 'Upper BB' and 'Lower BB' from the DataFrame
    hr = price_input_df.loc[price_input_df['Type'] == 'Hour', 'Value'].values
    prev_closed_price = price_input_df.loc[price_input_df['Type'] == 'Previous hr closed price', 'Value'].values
    closed_price = price_input_df.loc[price_input_df['Type'] == 'Current hr closed price', 'Value'].values
    high = price_input_df.loc[price_input_df['Type'] == 'high price', 'Value'].values
    low = price_input_df.loc[price_input_df['Type'] == 'low price', 'Value'].values
    upper_bb = price_input_df.loc[price_input_df['Type'] == 'Upper BB', 'Value'].values
    lower_bb = price_input_df.loc[price_input_df['Type'] == 'Lower BB', 'Value'].values
    MA20 = price_input_df.loc[price_input_df['Type'] == 'MA20', 'Value'].values
    MA50 = price_input_df.loc[price_input_df['Type'] == 'MA50', 'Value'].values
    MA100 = price_input_df.loc[price_input_df['Type'] == 'MA100', 'Value'].values

    if MA20.size > 0:
        print(MA20[0])
  
    #  # Check if both values are available before calculating the difference
    if  hr.size >0 and prev_closed_price.size > 0 and closed_price.size > 0 and upper_bb.size > 0 and lower_bb.size > 0 and MA20.size > 0 and MA50.size > 0 \
        and MA100.size > 0 and high.size >0 and low.size > 0  and upper_bb > 0 and lower_bb > 0:
        
        close_ma20_diff = closed_price[0] - MA20[0]
        close_ma50_diff = closed_price[0] - MA50[0]
        close_ma100_diff = closed_price[0] - MA100[0]
        up_bb20_low_bb20_diff = upper_bb[0] - lower_bb[0]
        ma50_ma100_diff = MA50[0] - MA100[0]
        upper_lower_diff = upper_bb[0] - lower_bb[0]
        
        market_state_lag1 = market_state_labelling(prev_closed_price[0], closed_price[0])
        bb_status = bb_status_labelling( closed_price[0], high[0], low[0], upper_bb[0], lower_bb[0] )
        
        
        # Display results
        st.write(f"Close - MA20 difference: {close_ma20_diff}")
        st.write(f"Close - MA50 difference: {close_ma50_diff}")
        st.write(f"Close - MA100 difference: {close_ma100_diff}")
        st.write(f"Upper BB - Lower BB difference: {up_bb20_low_bb20_diff}")
        st.write(f"MA50 - MA100 difference: {ma50_ma100_diff}")
        st.write(f"the status of bb is: {bb_status}")
        st.write(f"This is the distance between Upper BB and Lower BB: {upper_lower_diff}")
        st.write(f"Market_state: {market_state_lag1}")
        
        column_headers = ['close_ma20_diff','close_ma100_diff', 'up_bb20_low_bb20_diff', 'ma50_ma100_diff', 'hr', 'bb_status', 'market_state_lag1']
        user_input = [close_ma20_diff,close_ma100_diff, up_bb20_low_bb20_diff, ma50_ma100_diff, hr, bb_status, market_state_lag1]

        # Create DataFrame
        df = pd.DataFrame([user_input], columns=column_headers)
        print(df)
        return df

    else:
        st.write("Please make sure you entered all values.")


 



     