import os
import requests
import psycopg2



# Функция для получения секретов из Vault
def get_vault_secrets():
    VAULT_ADDR = os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200')
    VAULT_TOKEN = os.getenv('VAULT_TOKEN', 'root')

    url = f"{VAULT_ADDR}/v1/secret/data/db"
    headers = {'X-Vault-Token': VAULT_TOKEN}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()['data']['data']
        return {
            'user': data['POSTGRES_USER'],
            'password': data['POSTGRES_PASSWORD'],
            'dbname': data['POSTGRES_DB'],
        }
    else:
        print("Ошибка при получении секретов из Vault")
        return None