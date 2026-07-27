import pandas as pd
import numpy as np

def interpolate(df):

    # Load data
    df = df.sort_values('year').reset_index(drop=True)

    # --- 1. Create fractional time for yearly data (midpoint between month 6 and 7)
    t_year = df['year'].values + 6.5 / 12

    # Keep forcing columns only
    forcing_cols = df.columns.drop('year')
    Y = df[forcing_cols].values

    # --- 2. Build monthly fractional time axis
    start_year = df['year'].min()
    end_year = df['year'].max()

    months = pd.date_range(
        f"{start_year}-01-01",
        f"{end_year}-12-01",
        freq='MS'
    )

    t_month = months.year + (months.month - 1) / 12

    # --- 3. Interpolate (no extrapolation)
    Y_monthly = np.full((len(t_month), Y.shape[1]), np.nan)

    valid = (t_month >= t_year.min()) & (t_month <= t_year.max())

    for i in range(Y.shape[1]):
        Y_monthly[valid, i] = np.interp(
            t_month[valid],
            t_year,
            Y[:, i]
        )

    # --- 4. Build final DataFrame
    df_monthly = pd.DataFrame(Y_monthly, columns=forcing_cols)

    df_monthly.insert(0, 'time', months.strftime('%Y-%m-01'))
    df_monthly=df_monthly.dropna()

    return df_monthly

if __name__=="__main__":
    df=pd.read_csv('data_preparation/AR6_ERF_1750-2019.csv')
    df_monthly=interpolate(df)
    df_monthly = df_monthly[df_monthly['time'].between('1750-08-01', '2019-07-01')]
    df_monthly.to_csv('interpolatedAllForcing.csv')

