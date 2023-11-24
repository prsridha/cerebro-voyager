import re
import os
import json
import zipfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from kubernetes import client, config

from flask_cors import CORS
from flask import Flask, request, g

from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger

logging = CerebroLogger("dispatcher")
logger = logging.create_logger("server")

app = Flask(__name__)
CORS(app)


def run(cmd, shell=True, capture_output=True, text=True, haltException=True):
    try:
        out = subprocess.run(cmd, shell=shell, capture_output=capture_output, text=text)
        if out.stderr:
            if haltException:
                raise Exception("Command Error:" + str(out.stderr))
            else:
                print("Command Error:" + str(out.stderr))
        if capture_output:
            return out.stdout.rstrip("\n")
        else:
            return None
    except Exception as e:
        print("Command Unsuccessful:", cmd)
        print(str(e))
        raise Exception


def copy_files_to_pods(cli, root_path, cerebro_info):
    code_from_path = os.path.join(root_path, "code")
    code_to_path = cerebro_info["user_repo_path"]

    # get controller pod name
    # config.load_incluster_config()
    config.load_kube_config()
    v1 = client.CoreV1Api()
    cm1 = v1.read_namespaced_config_map(name='cerebro-info', namespace='cerebro')
    namespace = "cerebro"
    label = "app=cerebro-controller"

    pods_list = v1.list_namespaced_pod(namespace, label_selector=label, watch=False)
    controller_pod = pods_list.items[0].metadata.name

    # remove existing files in code_to_path
    cmd1 = "kubectl exec -t {} -c {} -- bash -c 'rm -rf {}/*' "
    run(cmd1.format(controller_pod, "cerebro-controller-container", code_to_path))

    # copy blank cerebro experiment file to user repo folder
    cmd2 = "kubectl cp -c cerebro-controller-container misc/CerebroExperiment.ipynb {}:{} --no-preserve"
    run(cmd2.format(controller_pod, code_to_path))

    if cli:
        return

    cmd3 = "kubectl cp -c {} {} {}:{} --no-preserve"
    cmd4 = "kubectl exec -t {} -c {} -- bash -c 'mv {}/code/* {}' "
    cmd5 = "kubectl exec -t {} -c {} -- bash -c 'rm -rf {}/code' "

    # copy to pods
    run(cmd3.format("cerebro-controller-container", code_from_path, controller_pod, code_to_path))

    # move and delete extra dir
    run(cmd4.format(controller_pod, "cerebro-controller-container", code_to_path, code_to_path))
    run(cmd5.format(controller_pod, "cerebro-controller-container", code_to_path))

    logger.info("Copied code files to pods")


def add_s3_credentials(s3_url):
    bucket_name = urlparse(s3_url, allow_fragments=False).netloc

    with open("misc/iam-policy-eks-s3.json", "r+") as f:
        policy = f.read()
        policy = policy.replace("<s3Bucket>", bucket_name)

        f.seek(0)
        f.write(policy)
        f.truncate()

    policy_arn_cmd = "aws iam list-policies --query 'Policies[?PolicyName==`{}`].Arn' --output text".format("eks-s3-policy")
    policy_arn = run(policy_arn_cmd)

    if "eks-s3-policy" in policy_arn:
        cmd = "aws sts get-caller-identity"
        account_id = json.loads(run(cmd))["Account"]
        arn = "arn:aws:iam::{}:policy/eks-s3-policy".format(account_id)
        detach_cmds = [
            "aws iam detach-role-policy --role-name {} --policy-arn {}".format("eks-s3-role",arn),
            "aws iam delete-policy --policy-arn {}".format(arn)
        ]
        for cmd in detach_cmds:
            run(cmd)
        print("Deleted S3 policy")

    cmd = """
    aws iam create-policy \
    --policy-name eks-s3-policy \
    --policy-document file://misc/iam-policy-eks-s3.json
    """
    run(cmd)
    logger.info("Created IAM read-only policy for S3")

    policy_arn = run(policy_arn_cmd)

    attach_cmd = "aws iam attach-role-policy --role-name {} --policy-arn {}".format("eks-s3-role", policy_arn)
    run(attach_cmd)
    logger.info("Attached IAM role to IAM policy for S3")


@app.before_request
def setup_context():
    g.root_path = "/data"

    # load values from cerebro-info configmap
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    cm = v1.read_namespaced_config_map(name='cerebro-info', namespace='cerebro')
    g.cerebro_info = json.loads(cm.data["data"])

    # create Key Value Store handle and initialize tables
    g.kvs = KeyValueStore(init_tables=True)
    logger.info("Initialized cerebro backend")


@app.route("/params", methods=["POST"])
def saveParams():
    params = request.json

    logger.info("Saving dataset locators in KVS")
    # save dataset locators on KVS
    g.kvs.set_dataset_locators(params)

    # add credentials for user's s3 bucket from params
    s3_url_pattern = r"'s3:\/\/.*?'"
    match = re.search(s3_url_pattern, str(params))
    s3_url = match.group().replace("'", "") if match else None
    add_s3_credentials(s3_url)

    resp = {
        "message": "Saved params json file",
        "status": 200
    }
    return resp


@app.route("/save-code/<route>", methods=["POST"])
def saveCode(route):
    # save zip file
    filename = "code.zip"
    file = request.files['file']
    file_path = os.path.join(g.root_path, filename)
    file.save(file_path)

    # extract zip file contents
    extract_path = os.path.join(g.root_path, "code")
    Path(extract_path).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(file_path, "r") as f:
        f.extractall(extract_path)
    os.remove(file_path)

    # check if the post request has the file part
    cli = route == "cli"
    copy_files_to_pods(cli, g.root_path, g.cerebro_info)

    resp = {
        "message": "Extracted and saved code zip file",
        "status": 200
    }

    return resp


@app.route("/get-urls", methods=["GET"])
def getURLs():
    public_dns = g.cerebro_info["public_dns_name"]

    # get jupyter string
    j = g.cerebro_info["jupyter_token_string"]
    j_bin = j.encode("utf-8")
    jToken = j_bin.hex().upper()

    jupyterP = g.cerebro_info["jupyter_node_port"]
    tensorboardP = g.cerebro_info["tensorboard_node_port"]
    grafanaP = g.cerebro_info["grafana_node_port"]
    # prometheusP = .cerebro_info["prometheus_node_port"]
    # lokiP = .cerebro_info["loki_port"]

    jupyterURL = "http://" + public_dns + ":" + str(jupyterP) + "/?token=" + jToken
    tensorboardURL = "http://" + public_dns + ":" + str(tensorboardP)
    grafanaURL = "http://" + public_dns + ":" + str(grafanaP)
    # prometheusURL = "http://" + public_dns + ":" + str(prometheusP)
    # lokiURL = "http://loki" + ":" + str(lokiP)

    message = {
        "jupyterURL": jupyterURL,
        "tensorboardURL": tensorboardURL,
        "grafanaURL": grafanaURL,
        # "prometheusURL": prometheusURL,
        # "lokiURL": lokiURL
    }

    return {
        "message": message,
        "status": 200
    }


@app.route("/health", methods=["GET"])
def health_ping():
    resp = {
        "message": "Hello! This is the Cerebro backend webserver",
        "status": 200
    }
    return resp


if __name__ == '__main__':
    logger.info("Starting cerebro backend server...")