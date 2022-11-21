from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.edgemodifier import Label


def _extract():
    file_path = Variable.get("data_folder") + "/" + Variable.get("test_df")
    uinfo = Variable.get("user_info", deserialize_json=True)
    print(uinfo, file_path)
    print(uinfo["uname"], uinfo["password"])


def _extract2(uname):
    print(f"Username: {uname}")


def _extract_env():
    print(Variable.get("user_info2", deserialize_json=True))


def _update_counter():
    counter = int(Variable.get("my_counter"))
    counter += 1
    Variable.set("my_counter", counter)


with DAG(dag_id="my_dag", description="DAG for showing nothing.",
         start_date=datetime(2021, 1, 1), schedule_interval=timedelta(minutes=5),
         dagrun_timeout=timedelta(minutes=10), tags=["learning_dag"], catchup=False) as dag:

    extract = PythonOperator(
        task_id="extract",
        python_callable=_extract
    )

    extract2 = PythonOperator(
        task_id="extract2",
        python_callable=_extract2,
        op_args=["{{ var.json.user_info.uname}}"])

    update = PythonOperator(
        task_id="update_counter",
        python_callable=_update_counter
    )

    # extract_env = PythonOperator(
    #     task_id="extract_env",
    #     python_callable=_extract_env
    # )

    # specify the order with ">>" and "<<" or ".setdownstream()" and ".setupstream()"
    extract >> Label("then run this") >> extract2 >> Label(
        "and then this") >> update
