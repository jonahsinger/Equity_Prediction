


RANDOM_SEED = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10
def load_and_prepare_data(file_path="data.csv"):
    # Load the data
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values detected:\n{missing[missing > 0]}")
        print("Dropping rows with missing values...")
        df = df.dropna()
    
    return df


def simple_train_test_split(df, test_size=TEST_SIZE):
    df = df.sort_index()
    
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
        
    X_train = train.drop('y', axis=1)
    y_train = train['y']
    X_test = test.drop('y', axis=1)
    y_test = test['y']
    
    return X_train, X_test, y_train, y_test