from ucimlrepo import fetch_ucirepo
import pandas as pd  

def download():
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets

    df = pd.concat([X,y], axis=1)

    df.to_csv('app\\data\\data.csv', index=None)

if __name__ == '__main__':
    download()