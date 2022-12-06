# Notes
Doesn't run well against Windows, but can work well from WSL2, using the [instructions from AWS](https://aws.amazon.com/blogs/big-data/introducing-aws-glue-interactive-sessions-for-jupyter/)

For file performance, remember not to have the venv in `/mnt/c/...` but run in the Linux system `~/projects/...`; so navigate there to create the venv -- I had to manually specify the interpreter path there as `/home/user/projects/glue_studio/venv/bin/python`.

## Install instruction changes
I had to change the following to get it to work:
1. `pip3 install --user --upgrade jupyter boto3 aws-glue-sessions` wouldn't run with `--user` inside a WSL2 venv; so had to install globally (in the venv)
1. `jupyter kernelspec install $SITE_PACKAGES/aws_glue_interactive_sessions_kernel/glue_pyspark` and `jupyter kernelspec install $SITE_PACKAGES/aws_glue_interactive_sessions_kernel/glue_spark` wouldn't install globally, I had to add `--user` to them
1. `aws iam attach-user-policy --role-name <myIAMUser> --policy-arn arn:aws:iam::aws:policy/AWSGlueConsoleFullAccess` wasn't working; I edited `~/.aws/config` manually and added it manually (see below)


### Sample ~/.aws/config
```
[default]
region = eu-west-1
output = json
glue_role_arn=arn:aws:iam::336785731537:role/Glue_studio_role
```

## Launch script
Use the following launch script to always re-open the folder in WSL: `code --remote wsl+Ubuntu --folder-uri "vscode-remote://wsl+Ubuntu/mnt/c/Projects/Glue Studio"`.  
Then launch Jupyter with `jupyter notebook`.  
*NB, the browser doesn't auto-redirect; so click the link with the token to open the notebook*

## Run Notebook from VS Code
Or open the notebook file directly in VS Code, select `Connect to Jupyter Server` and paste the URL (incl token), then select kernel (e.g. `Glue Pyspark`).  

## Avoid linter warnings
To avoid linter warnings about missing imports, install these dependencies as well:
- pyspark
- fake-awsglue