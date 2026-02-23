import pandas as pd


def main():
    # Read input with missing values
    df = pd.read_csv("test1.csv")

    # 1.1 Covariance: skip missing rows
    cov_skip = df.dropna(axis=0, how="any").cov()
    cov_skip.to_csv("testout_1.1.csv", index=False)

    # 1.2 Correlation: skip missing rows 
    corr_skip = df.dropna(axis=0, how="any").corr()
    corr_skip.to_csv("testout_1.2.csv", index=False)

    # 1.3 Covariance: pairwise deletion 
    cov_pairwise = df.cov()
    cov_pairwise.to_csv("testout_1.3.csv", index=False)

    # 1.4 Correlation: pairwise deletion 
    corr_pairwise = df.corr()
    corr_pairwise.to_csv("testout_1.4.csv", index=False)


if __name__ == "__main__":
    main()
