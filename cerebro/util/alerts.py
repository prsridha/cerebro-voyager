# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from IPython.display import display, HTML


def filter_traceback(error_message):
    modules = [
        "etl_controller.py",
        "etl_worker.py",
        "experiment.py",
        "kvs",
        "cerebro"
    ]

    tb = error_message.splitlines()
    filtered_tb = []
    for line in tb:
        if not any(module in line for module in modules):
            filtered_tb.append(line)

    return "<br>".join(filtered_tb)


def html_alert(error_message):
    filtered_message = filter_traceback(error_message)
    error_message = f"An error occurred: \n {filtered_message}"
    display(HTML(f'<div class="alert alert-danger">{error_message}</div>'))
    return True
