import time


class TimedList:
    """
    The items in the list will expire and been removed after the defined time. This will be used in reward processing
    to prevent the same reward being given multiple times in multiple frames.
    """

    def __init__(self, lifespan=5):
        self.lifespan = lifespan

        self.list = {}  # {item: expiry time}

    def __contains__(self, key):
        if key in self.list.keys():
            if self.list[key] > time.time():
                return True
            else:
                del self.list[key]
                return False
        else:
            return False

    def add(self, item, timestamp):
        self.list[item] = timestamp + self.lifespan
