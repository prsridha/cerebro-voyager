from IPython.display import display, HTML


def filter_traceback(error_message):
    modules = [
        "etl_controller.py",
        "etl_worker.py",
        "experiment.py"
    ]

    tb = error_message.splitlines()
    filtered_tb = []
    for line in tb:
        if not any(module in line for module in modules):
            filtered_tb.append(line)

    return "\n".join(filtered_tb)


def html_alert(error_message):
    filtered_message = filter_traceback(error_message)
    error_message = f"An error occurred: \n {filtered_message}"
    display(HTML(f'<div class="alert alert-danger">{error_message}</div>'))
    return True
