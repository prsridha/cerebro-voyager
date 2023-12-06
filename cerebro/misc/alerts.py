from IPython.display import display, HTML


def html_alert(error_message):
    error_message = f"An error occurred: {str(error_message)}"
    display(HTML(f'<div class="alert alert-danger">{error_message}</div>'))
    return True
