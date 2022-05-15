import time


def print_frame_rate(time_stamp_deque, name="Undefined"):
    """
    This needs to be used in a thread.
    """
    while True:
        mean_time = (time_stamp_deque[-1] - time_stamp_deque[0]) / (time_stamp_deque.maxlen - 1)
        print("\n{} frame rate: {}".format(name, 1 / mean_time))
        time.sleep(10)
