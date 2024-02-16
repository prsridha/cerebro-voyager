<h1>Cluster Setup Options</h1>
The full list of supported <code>cerebro.py</code> command options are given below. This has to be executed from the cerebro-voyager.py directory.

1. <b>start</b>: this command starts your Cerebro instances on the cluster. It takes in an optional 'workers' argument, which is the number of worker nodes. If not explicitly mentioned, this parameter defaults to the last value used. Once Cerebro instances are created, it outputs the SSH tunneling command and the links to JupyterNotebook and Tensorboard. 

    Usage -
    ```
    python3 cerebro.py start --workers 4
    ```

2. <b>shutdown</b>: this will release all allocated Cerebro resources and delete any associated files. If there are a large number of downloaded files, it might take a while to clear the saved files.

    Usage -
    ```
    python3 cerebro.py shutdown
    ```

3. <b>restart</b>: this is equivalent to running the <i>shutdown</i> and <i>start</i> commands sequentially.

    Usage -
   ```
   python3 cerebro.py restart
   ```

4. <b>url</b>: this will output the SSH tunnel command, along with the JupyterNotebook and the Tensorboard URLs of an existing Cerebro experiment. This can be used if the terminal output from the start command has been cleared.

    Usage -  
    ```
    python3 cerebro.py url
    ```