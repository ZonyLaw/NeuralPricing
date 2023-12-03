import streamlit as st


def predict_binary_output(X_live, model_features, churn_pipeline_dc_fe, churn_pipeline_model):

    # from live data, subset features related to this pipeline
    X_live_subset = X_live.filter(model_features)

    # apply data cleaning / feat engine pipeline to live data
    X_live_subset_dc_fe = churn_pipeline_dc_fe.transform(X_live_subset)

    # predict
    model_prediction = churn_pipeline_model.predict(X_live_subset_dc_fe)
    model_prediction_proba = churn_pipeline_model.predict_proba(
        X_live_subset_dc_fe)
    # st.write(model_prediction_proba)

    # Create a logic to display the results
    result_prob = model_prediction_proba[0, model_prediction][0]*100
    if model_prediction == 1:
        model_results = 'buy'
    else:
        model_results = 'sell'

    statement = (
        f'### There is {result_prob.round(1)}% probability '
        f'that this is a **{model_results} position**.')

    st.write(statement)

    return model_prediction


def predict_profits(X_live, model_feature, model_pipeline, model_label_map):

    category_phrase = {
        '-25<': 'Sell with >25 pips target',
        '-20': 'Sell with 20 pips target',
        '-5': 'Sell with 5 pips target',
        '5': 'Buy with 5 pips target',
        '20': 'Buy with 20 pips target',
        '>25': 'Buy with >25 pips target'
    }
    
    # from live data, subset features related to this pipeline
    X_live_subset = X_live.filter(model_feature)
    
    # predict
    model_prediction = model_pipeline.predict(X_live_subset)
    model_prediction_proba = model_pipeline.predict_proba(X_live_subset)
    # st.write(model_prediction_proba)

    # create a logic to display the results
    proba = model_prediction_proba[0, model_prediction][0]*100
    category_labels = model_label_map[model_prediction[0]]


    # output the entire array    
    proba1 = model_prediction_proba[0, 0]*100
    category_labels1 = model_label_map[0]
    proba2 = model_prediction_proba[0, 1]*100
    category_labels2 = model_label_map[1]
    proba3 = model_prediction_proba[0, 2]*100
    category_labels3 = model_label_map[2]
    proba4 = model_prediction_proba[0, 3]*100
    category_labels4 = model_label_map[3]
    proba5 = model_prediction_proba[0, 4]*100
    category_labels5 = model_label_map[4]
    proba6 = model_prediction_proba[0, 5]*100
    category_labels6 = model_label_map[5]

   
    print( model_prediction_proba[0,0]*100)

    
    statement = (
        f"* If you are in this trade, there is a {proba.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels]} pips**.\n\n"
        
        f"-------------------------------All statistics-------------------------------\n\n"
        
        f"* If you are in this trade, there is a {proba1.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels1]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba2.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels2]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba3.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels3]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba4.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels4]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba5.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels5]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba6.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels6]} pips**.\n\n"

    )
    

    st.write(statement)


def predict_cluster(X_live, cluster_features, cluster_pipeline, cluster_profile):

    # from live data, subset features related to this pipeline
    X_live_cluster = X_live.filter(cluster_features)

    # predict
    cluster_prediction = cluster_pipeline.predict(X_live_cluster)

    statement = (
        f"### The prospect is expected to belong to **cluster {cluster_prediction[0]}**")
    st.write("---")
    st.write(statement)

  	# text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    statement = (
        f"* Historically, **users in Clusters 0  don't tend to Churn** "
        f"whereas in **Cluster 1 a third of users churned** "
        f"and in **Cluster 2 a quarter of users churned**."
    )
    st.info(statement)

  	# text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    statement = (
        f"* The cluster profile interpretation allowed us to label the cluster in the following fashion:\n"
        f"* Cluster 0 has user without internet, who is a low spender with phone\n"
        f"* Cluster 1 has user with Internet, who is a high spender with phone\n"
        f"* Cluster 2 has user with Internet , who is a mid spender without phone"
    )
    st.success(statement)

    # hack to not display index in st.table() or st.write()
    cluster_profile.index = [" "] * len(cluster_profile)
    # display cluster profile in a table - it is better than in st.write()
    st.table(cluster_profile)
