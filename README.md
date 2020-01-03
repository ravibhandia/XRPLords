# XRP Wallet Discovery 

XRP transactions occur with variance in terms of fees and network speeds. Being able to identify types of transactions and cluster those from others may provide insight to allow the development of niche optimization algorithms. **Other background / high-level view...**

[*... more explanation needed...*]

The purpose of this project is to classify X number of transactions from ...  

## Directory Structure 

**Data/** contains results from SQL commands on XRP database 

**Notebooks/** contains Jupyter Notebooks for data exploration and analysis 

**Utils/** contains *.py* functions used throughout our XRP adventures

E.g. 

```python
        from Utils.bigquery_utils import *
        
        key_path = "../credential/Xpring...json"
        cred = bigquery_utils.get_gcp_creds(key_path)
        
        # SQL query string
        start_date = "2019-12-19"
        end_date = "2019-12-26"
        sql_query = bigquery_utils.get_sql_query(start_date, end_date)
        
        # Get dataframe from GCP query
        gcp_dataframe = gqp_query(sql_query, cred = cred)
```