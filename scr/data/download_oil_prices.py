import lineapy
import pandas as pd 
import requests
import re 

def download_and_tidy_oil_prices(url):
    
    response = requests.get(url)
    df = pd.read_excel(
        response.content,
        sheet_name= "Data 12",
        index_col=0,
        skiprows=2, 
        parse_dates=["Date"], 
        ).rename(
            columns = lambda c: re.sub(
                "\(PADD 1[A-C]\)",
                "",
                c.replace("Weekly ", "").replace(
                    " All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)",
                    "",

                ),

            ).strip()       
        
    )

    df.to_csv("data/processed/weekly_gas_price_data.csv", index=False)

    # lineapy.save(df, "../data/processed/weekly_gas_price_data")

    return df 


if __name__ == '__main__':
    download_and_tidy_oil_prices("https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls")