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

def gpq_query(sql_query, cred, query_params = None):
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
        pd_query_result = client.query(sql_query, job_config = job_config).to_dataframe()
    except:
        pd_query_result = None;
    return pd_query_result;
