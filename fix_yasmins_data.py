
if __name__ == "__main__":
    file = open("portfolio_data_xgboost_500.txt", 'r')
    out = open("portfolio_data_xgboost_500_clean.txt", 'w')

    for line in file:
        split = line.split('!')
        corr = ((split[0])[1:])[:-1]
        out.write(corr + '!' + split[1] + '\n')
    file.close()
    out.close()