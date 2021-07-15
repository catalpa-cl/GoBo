

def display_loading_bar(step, total_steps, verbose=False, bar_size=30):
    """
    Display a loading bar in the console.

    :param step: The current step.
    :param total_steps: The total number of steps.
    :param verbose: Whether or not to display the current step index.
    :param bar_size: The size of the loading bar.
    :return: Returns the loading bar as a string.
    """

    s = '['
    percent = int(step / (total_steps - 1) * bar_size)

    for i in range(bar_size):
        if i < percent:
            s += '='
        elif i > percent:
            s += ' '
        else:
            s += '>'

    return s + ']' + (' - ' + str(step) + ' / ' + str(total_steps) if verbose else '')
