import pathlib
import pickle

import gas_price_forecast_module
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


def task_weekly_gas_price_data():
    df = gas_price_forecast_module.get_weekly_gas_price_data()

    if not pathlib.Path("/tmp").joinpath("gas_price_forecast").exists():
        pathlib.Path("/tmp").joinpath("gas_price_forecast").mkdir()
    pickle.dump(df, open("/tmp/gas_price_forecast/variable_df.pickle", "wb"))


def task_weekly_gas_price_data_long():
    df = pickle.load(open("/tmp/gas_price_forecast/variable_df.pickle", "rb"))

    df_long = gas_price_forecast_module.get_weekly_gas_price_data_long(df)

    if not pathlib.Path("/tmp").joinpath("gas_price_forecast").exists():
        pathlib.Path("/tmp").joinpath("gas_price_forecast").mkdir()
    pickle.dump(df_long, open("/tmp/gas_price_forecast/variable_df_long.pickle", "wb"))


def task_gas_price_forecast(cutoff_date, region):
    cutoff_date = str(cutoff_date)

    region = str(region)

    df_long = pickle.load(open("/tmp/gas_price_forecast/variable_df_long.pickle", "rb"))

    full_plot = gas_price_forecast_module.get_gas_price_forecast(
        cutoff_date, df_long, region
    )

    if not pathlib.Path("/tmp").joinpath("gas_price_forecast").exists():
        pathlib.Path("/tmp").joinpath("gas_price_forecast").mkdir()
    pickle.dump(
        full_plot, open("/tmp/gas_price_forecast/variable_full_plot.pickle", "wb")
    )


def task_setup():
    pickle_folder = pathlib.Path("/tmp").joinpath("gas_price_forecast")
    if not pickle_folder.exists():
        pickle_folder.mkdir()


def task_teardown():
    pickle_files = pathlib.Path("/tmp").joinpath("gas_price_forecast").glob("*.pickle")
    for f in pickle_files:
        f.unlink()


default_dag_args = {
    "owner": "airflow",
    "retries": 2,
    "start_date": days_ago(1),
    "params": {"region": "U.S.", "cutoff_date": "2023-04-16"},
}

with DAG(
    dag_id="gas_price_forecast_dag",
    schedule_interval="*/15 * * * *",
    max_active_runs=1,
    catchup=False,
    default_args=default_dag_args,
) as dag:
    weekly_gas_price_data = PythonOperator(
        task_id="weekly_gas_price_data_task",
        python_callable=task_weekly_gas_price_data,
    )

    weekly_gas_price_data_long = PythonOperator(
        task_id="weekly_gas_price_data_long_task",
        python_callable=task_weekly_gas_price_data_long,
    )

    gas_price_forecast = PythonOperator(
        task_id="gas_price_forecast_task",
        python_callable=task_gas_price_forecast,
        op_kwargs={
            "cutoff_date": "{{ params.cutoff_date }}",
            "region": "{{ params.region }}",
        },
    )

    setup = PythonOperator(
        task_id="setup_task",
        python_callable=task_setup,
    )

    teardown = PythonOperator(
        task_id="teardown_task",
        python_callable=task_teardown,
    )

    gas_price_forecast >> teardown

    setup >> weekly_gas_price_data

    weekly_gas_price_data >> weekly_gas_price_data_long

    weekly_gas_price_data_long >> gas_price_forecast
