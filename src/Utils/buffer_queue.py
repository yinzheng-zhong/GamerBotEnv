import multiprocessing.queues as mpq
import multiprocessing as mp


class BufferQueue(mpq.Queue):
    """
    multiprocessing queue subclass like deque


    NOT USED.
    """
    def __init__(self, *args, **kwargs):
        ctx = mp.get_context()
        super(BufferQueue, self).__init__(*args, **kwargs, ctx=ctx)

    def append(self, item):
        self.put(item)

    def queue(self):
        o = getattr(self, 'queue', None)[-1]
        return o
