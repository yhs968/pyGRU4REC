# 1. create an order for the sessions and sort

def time_sort(df, session_key, time_key):
    '''
    Reorder the df by time
    '''
    sess_orders = df.groupby(session_key)[time_key].min().sort_values().index
    df[session_key] = pd.Categorical(df[session_key], sess_orders)
    df.sort_values([session_key, time_key], inplace = True)

def get_sessions(df, session_key, item_key):
    '''
    Extract the items for each sessions from the df
    '''
    sessions = df.groupby(session_key)[item_key].apply(list).tolist()
    
    return sessions    
    
session_key = 'SessionId'
time_key = 'Time'
item_key = 'ItemId'

time_sort(df_valid, session_key, time_key)
sessions_valid = get_sessions(df_valid, item_key)
sessions_lengths = [len(sess) for sess in sessions_valid]