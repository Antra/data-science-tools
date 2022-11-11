# Notes
These are my notes from playing around with an Airflow setup for pipeline orchestration; following [Apache Airflow for Data Science - How to Install Airflow Locally](https://betterdatascience.com/apache-airflow-install/).

Remember that Airflow doens't work on Windows, so it needs to be done from WSL!

# WSL Setup
- Install WSL if it isn't already: [Install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
    - *Upgrade WSL kernel if needed: `wsl --update`*
- Launch WSL from a Windows terminal: `wsl`
- Update the WSL packages if they haven't been for a while: `sudo apt update && sudo apt upgrade`
- Navigate to the code folder (e.g. `cd ~/projects`)
    - NB! file system performance is much faster when using the native WSL filesystem in e.g. `~/projects` compared to the SMB mounted in e.g. `/mnt/c/projects`.
- Launch VS Code from the folder `code .`
    - *VS Code launches in the chosen WSL distro*
- Create a virtual environment as always: `python3 -m venv venv`
    - *select to keep using the virtual environment as default for this workspace*

Be aware that the Python version may differ from the normal system version!

# Dependencies
Installation depends both on Python and Airflow; so constraint files should be used for "known working combinations" (which Airflow maintains), see [https://github.com/apache/airflow#installing-from-pypi](Installing from PyPi).  
For example: `https://raw.githubusercontent.com/apache/airflow/constraints-2.4.2/constraints-no-providers-3.8.txt`, where `2.4.2` and `3.8` respectively refers to the Airflow and Python versions.

Ensure that the Airflow versions and Python version for the environment match and then install as normal from PyPi, for example:
```
python -m pip install "apache-airflow==2.4.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.4.2/constraints-no-providers-3.8.txt"
```


# Init
Initialise a new airflow database with `airflow db init`, this will create a new folder in %userprofile% (or ~/) called `airflow`.  
Change to it with `cd ~/airflow`.

The files in there are:
- `airflow.cfg` which contains the airflow config
- `airflow.db` which is the Metastore that Airflow uses

## User creation
Create a user with
```
airflow users create \ 
    --username admin \
    --password admin \
    --firstname <FirstName> \
    --lastname <LastName> \
    --role Admin \
    --email <YourEmail>
```

## Start Airflow Webserver and scheduler
Launch:
- Airflow Webserver `airflow webserver -D`
- Airflow Scheduler `airflow scheduler -D`

And then navigate to the [Airflow UI](http://localhost:8080)

### Launch script
Use a launch script like:
```
cd ~/airflow && rm -f *.pid && source ~/projects/data-science-tools/Airflow/venv/bin/activate && airflow webserver -D && airflow scheduler -D && deactivate
```

And schedule it from `cron` - or when using wsl schedule with the Windows Task scheduler as:
- Trigger: at log on
- Action: Start a program
- Program/Script: `C:\Windows\System32\wsl.exe`
- Add arguments: `-u <username> "/home/<username>/path/launch.sh"`