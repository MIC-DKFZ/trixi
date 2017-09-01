# Install
Install Dependencies
```
pip install numpy seaborn matplotlib
#install pytorch: http://pytorch.org/
pip install torchvision
```

Install vislogger
```
git clone https://phabricator.mitk.org/source/vislogger.git
cd vislogger
pip install -e .
```

# Use on remote server

If you want to run vislogger on a remote server, but show the reuslts locally
you can do:

```
# On local computer:
ssh -N -f -L localhost:8099:localhost:8097 USERNAME@REMOTE_SERVERNAME


python -m visdom.server
python my_random_vislogger_script.py
```

Now on your local computer you can go to `localhost:8099` and see the visdom dashboard.