def data_preprocessing(model):
    import pandas as pd
    import numpy as np

    if model == "train":
        c = pd.read_csv(r"training_data.csv")
        d = c[(c['transaction_date'] == c['last_retry_date']) | (c['retries'] == -1)]
        d['dunning_cnt'] = d['no_of_times_dunning_to_existing'] + d['no_of_times_dunning_to_churned']
        d['cust_since_cnt'] = d['no_of_times_dunning_to_churned'] + d['no_of_successful_payments_by_account']
        # getting probability of dunning
        d['dunning%'] = ((d['dunning_cnt'] / d['cust_since_cnt']) * 100)
        d['y_label'] = np.where((d['dunning%'] < 30) | (d['has_payment_method_updated'] == 'Yes'), 0, 1)
        d['dunning%'].fillna((d['dunning%'].median()), inplace=True)
        d['payment_update'] = np.where(d['has_payment_method_updated'] == 'Yes', 1, 0)
        d['payment_update'].value_counts()

        X_data = d[['no_of_times_dunning_to_churned', 'no_of_successful_payments_by_order',
                    'no_of_successful_payments_by_account',
                    'dunning_cnt', 'cust_since_cnt', 'dunning%', 'y_label', 'payment_update']]
        X_data.dropna(how='any', inplace=True)
        X_train = X_data[['no_of_times_dunning_to_churned', 'no_of_successful_payments_by_order',
                          'no_of_successful_payments_by_account',
                          'dunning_cnt', 'cust_since_cnt', 'dunning%', 'payment_update']]
        Y_data = X_data[['y_label']]

        return X_train, Y_data
		
    if model == "predict":
        pred_data = pd.read_csv(r"Predicted_data.csv")

        pred_data['dunning_cnt'] = pred_data['no_of_times_dunning_to_existing'] + pred_data['no_of_times_dunning_to_churned']
        pred_data['cust_since_cnt'] = pred_data['no_of_times_dunning_to_churned'] + pred_data['no_of_successful_payments_by_account']
        # getting probability of dunning
        pred_data['dunning%'] = ((pred_data['dunning_cnt'] / pred_data['cust_since_cnt']) * 100)
        pred_data['dunning%'].fillna((pred_data['dunning%'].median()), inplace=True)
        pred_data['payment_update'] = np.where(pred_data['has_payment_method_updated'] == 'Yes', 1, 0)
        pred_data['y_label'] = np.where((pred_data['dunning%'] < 30) | (pred_data['has_payment_method_updated'] == 'Yes'), 0, 1)

        CrossChk = pred_data[['no_of_times_dunning_to_churned','no_of_successful_payments_by_order', 'no_of_successful_payments_by_account','dunning_cnt', 'cust_since_cnt', 'dunning%', 'has_payment_method_updated']]
		
        return CrossChk, pred_data