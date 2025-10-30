# 0001 - Log service

Implement a log service that provides a public API with the following functionalities:

1. Shortlog : providing the most recent events by event type.
2. Replay : providing the entire log of events as received.
3. Ingest : Allows to add and ingest a new event in the logger service

The service should be thread safe, and sustain a high rate of adhoc writes. 

```python
import threading
from collections import deque
from typing import Dict, Any, Deque, Tuple
from contextlib import contextmanager
from typing import Iterator

class rw_lock:

    #----------------------------------------------------------------------
    # Implementation of a read-write lock, allows to operate on a resource
    # asymmetrically while reading and writing. Readers to the resource
    # are tracked by a counter allowing multiple readers at the same time. While
    # write access to the resource is guaranteed to be single threaded. Writes
    # cannot happen at the same time with writes. 
    # 
    # This ensures reads don't block other reads and writes exclusively modify
    # the protected resource.
    #----------------------------------------------------------------------


    def __init__(self):
        self._ready = threading.Condition()
        self._writer = False
        self._readers = 0

    def acquire_read(self):
        """
        Ensure reads happen only when no write access by another thread
        """
        with self._ready:
            while self._writer:
                self._ready.wait()
            self._readers += 1

    def release_read(self):
        """
        Make sure previously acquired lock is released when no more threads
        """
        with self._ready:
            self._readers -=1
            if self._readers == 0:
                self._ready.notify_all()


    def acquire_write(self):
        """
        Ensure access during write only by a single thread to modify
        """
        with self._ready:
            while self._readers > 0:
                self._ready.wait()
            self._writer = True

    def release_write(self):
        """
        Make sure previously acquired write access lock is released
        """
        with self._ready:
            self._writer = False
            self._ready.notify_all()


    @contextmanager
    def read(self):
        """
        Manage context for reading
        """
        try:
            self.acquire_read()
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self):
        """
        Manage context for writing
        """
        try:
            self.acquire_write()
            yield
        finally:
            self.release_write()


class logger_service:
    #----------------------------------------------------------------------
    # In memory log service 
    #
    # Provides public API with the following functionality:
    #   1. Ingest a record 
    #   2. Replay the records as they arrived
    #   3. Retrieve a shortlog with the most recent events for each event type
    # 
    # Operates by adding the received request soonest possible to the log,
    # and then keeping the most recent event per type in dictionary. Access to
    # the public API is guaranteed thread safe. The are two locks protecting 
    # each of the two relevant resources.
    #
    # It supports snapshoting that guarantees reliability and consistent data
    # up until the point of handling the request.
    #----------------------------------------------------------------------


    def __init__(self):
        self._raw_lock : rw_lock = rw_lock()
        self._raw_copies = 32
        self._raw : Deque[Tuple[int, str, Any]] = deque()

        self._latest_lock : rw_lock = rw_lock()
        self._latest_copies = 64
        self._latest : Dict[str,Tuple[Any, int]] = {}


    def ingest(self, key: str, value: Any, timestamp: int):
        # Obtain the raw log write lock and append to the log
        with self._raw_lock.write():
            self._raw.append((timestamp, value, key))

        # Update the event's latest record
        with self._latest_lock.write():
            latest_event = self._latest.get(key,None)
            if latest_event is None:
                self._latest[key]=(value, timestamp)

            elif latest_event is not None \
                and latest_event[1] is not None \
                and latest_event[1] < timestamp:
                     
                self._latest[key] = (value, timestamp)


    def replay(self) -> Iterator[Tuple[int, str, Any]]:
        with self._raw_lock.read():
            if self._raw_copies > 0:
                raw_snapshot = [ e for e in self._raw ]
                self._raw_copies -= 1
            else:
                raise Exception("Too many raw copies")

        for ts, value, key in raw_snapshot:
            record = (ts, key, value)
            yield record

        with self._raw_lock.read():
            self._raw_copies +=1


    def shortlog(self) -> Iterator[Tuple[int, str, Any]]:
        with self._latest_lock.read():
            if self._latest_copies > 0:
                latest_snapshot = self._latest.copy()
                self._latest_copies -= 1
            else:
                raise Exception("Too many latest copies")

        for k, v in latest_snapshot.items():
            record = (v[1],k,v[0])
            yield record

        with self._latest_lock.read():
            self._latest_copies += 1  

```

