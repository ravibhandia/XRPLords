# Utill fns for bigquery 
from os import path

from google.cloud import bigquery
from google.oauth2 import service_account

def get_gcp_creds(key_path):
    """
    Return GCP credentials from local .json
    
    Parameters
    ----------
    key_path: str
        Path to local credential .json
    """
    try:
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        )
    except:
        credentials = None;
    return credentials;

<<<<<<< HEAD
def gbq_query(sql_query, cred, query_params = None):
=======
def gpq_query(sql_query, cred, query_params = None):
>>>>>>> 18a3010a22231928bc152eaf501c8fac9f0eae72
    """
    Run a query against BigQuery, returning Pandas dataframe
    
    Parameters
    ----------
    sql_query: str
        SQL query string 
    cred: obj
        Credential class instance 
    query_params: list, optionl 
        Query parameters to pass into the query string 
    """
    try:
        client = bigquery.Client(credentials = cred)
        job_config = bigquery.QueryJobConfig()
<<<<<<< HEAD
        job_config.query_parameters = query_params
=======
>>>>>>> 18a3010a22231928bc152eaf501c8fac9f0eae72
        pd_query_result = client.query(sql_query, job_config = job_config).to_dataframe()
    except:
        pd_query_result = None;
    return pd_query_result;
