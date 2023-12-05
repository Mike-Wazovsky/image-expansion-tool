# Model to prod tutorial

## Run local server

``` bash
cp .env.example .env
python app.py
```

## Run client

``` bash
python client.py
```

## Run production server

``` bash
zip -r model2prod.zip . -x venv\* -x .git\* -x .idea\* -x __pycache__\*
scp model2prod.zip user@server:projects/model2prod.zip
ssh user@server
cd projects
unzip model2prod.zip -d model2prod
cd model2prod
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
waitress-serve --port=12023 app:app
```

## Run in background

### 1. Tmux

It's a terminal multiplexer. It allows you to run multiple terminal sessions inside a single terminal window. It also
allows you to detach from a terminal session and reattach to it later.

``` bash
tmux new -s model2prod
cd projects/model2prod
source venv/bin/activate
waitress-serve --port=12023 app:app
```

Detach from the tmux session by pressing Ctrl+B followed by D. You can now log out of the server and your process will
keep running.
To reattach to the tmux session, use the following command:

``` bash
tmux attach -t model2prod
```

To close the tmux session, use the following command:

``` bash
tmux kill-session -t model2prod
```

To list all tmux sessions, use the following command:

``` bash
tmux ls
```

### 2. Screen

It's a terminal multiplexer. It allows you to run multiple terminal sessions inside a single terminal window. It also
allows you to detach from a terminal session and reattach to it later.

``` bash
screen -S model2prod
cd projects/model2prod
source venv/bin/activate
waitress-serve --port=12023 app:app
```

Detach from the screen session by pressing Ctrl+A followed by D. You can now log out of the server and your process will
keep running.
To reattach to the screen session, use the following command:

``` bash
screen -r model2prod
```

### 3. nohup

It's a command that allows you to run a command or shell script that can continue running in the background after you
log out from a shell.

``` bash
cd projects/model2prod
source venv/bin/activate
nohup waitress-serve --port=12023 app:app &
```

To look at the output of the process, use the following command:

``` bash
cat nohup.out
```

To check the process id of the running process, use the following command:

``` bash
ps aux | grep waitress-serve
```

To kill the process, use the following command:

``` bash
kill -9 <process_id>
```

<details>
<summary>Advanced stuff</summary>

What if you want to run other server with waitress - you cant because name of the process is the same. You can use this
command to run named process (e.g. model2prod):

``` bash
nohup bash -c 'exec -a model2prod waitress-serve --port=12023 app:app' &
```

To kill the process, use the following command:

``` bash
pkill -f model2prod
```

</details>

## Systemd

It's a system and service manager for Linux operating systems. It's designed to start up and supervise system processes.
It's also designed to automatically restart processes in case of failure.

``` bash
cp conf/model2prod.service /etc/systemd/system/model2prod.service
sudo systemctl enable model2prod
sudo systemctl start model2prod
sudo systemctl status model2prod
```

To stop the service, use the following command:

``` bash
sudo systemctl stop model2prod
```

To restart the service, use the following command:

``` bash
sudo systemctl restart model2prod
```

To check the logs in real time, use the following command:

``` bash
sudo journalctl -u model2prod -f
```

## Supervisor

It's a client/server system that allows its users to monitor and control a number of processes on UNIX-like operating
systems.

``` bash
cp conf/model2prod.conf /etc/supervisor/conf.d/model2prod.conf
sudo mkdir -p /var/log/model2prod
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start model2prod
sudo supervisorctl status model2prod
```

To stop the service, use the following command:

``` bash
sudo supervisorctl stop model2prod
```

To restart the service, use the following command:

``` bash
sudo supervisorctl restart model2prod
```

To check the logs in real time, use the following command:

``` bash
sudo tail -f /var/log/model2prod/model2prod.*.log
```

## Docker

It's a tool designed to make it easier to create, deploy, and run applications by using containers.

``` bash
docker build -t model2prod .
docker run -d -p 12023:12023 --name model2prod --restart unless-stopped --env-file .env model2prod
docker ps
docker logs model2prod
```

To stop the service, use the following command:

``` bash
docker stop model2pr
```

<details>
<summary>Container name is already in use</summary>

If you get the following error:

`Error response from daemon: Conflict. The container name "/model2prod" is already in use by container`

You can use the following command to remove the container:

``` bash
docker rm model2prod
```

</details>


To restart the service, use the following command:

``` bash
docker restart model2prod
```

To check the logs in real time, use the following command:

``` bash
docker logs -f model2prod
```

<details>
<summary>Really advanced stuff</summary>

# Deploying a Docker container to a remote server

## Manual

``` bash
docker build -t model2prod .
docker save -o model2prod.tar model2prod
scp model2prod.tar user@server:projects/model2prod/model2prod.tar
scp .env user@server:projects/model2prod/.env
ssh user@server
cd projects/model2prod
docker load -i model2prod.tar
docker run -d -p 12023:12023 --name model2prod --restart unless-stopped --env-file .env model2prod
```

## Image registry

``` bash
docker build -t model2prod .
docker tag model2prod user/model2prod
docker push user/model2prod
ssh user@server
docker pull user/model2prod
docker run -d -p 12023:12023 --name model2prod --restart unless-stopped --env-file .env user/model2prod
```

</details>

# Further reading

- Nginx - https://www.nginx.com/resources/glossary/nginx/
- Ansible - https://www.ansible.com/resources/get-started
- Docker-compose - https://docs.docker.com/compose/
- CI/CD - https://www.atlassian.com/continuous-delivery/principles/continuous-integration-vs-delivery-vs-deployment
- Github Actions - https://docs.github.com/en/actions
- Monitoring - https://developer.nvidia.com/blog/a-guide-to-monitoring-machine-learning-models-in-production/
- Prometheus - https://prometheus.io/docs/introduction/overview/
- Grafana - https://grafana.com/docs/grafana/latest/getting-started/