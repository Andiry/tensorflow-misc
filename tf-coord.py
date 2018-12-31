import tensorflow as tf
import numpy as np
import threading
import time

def myloop(coord, worker_id):
  while not coord.should_stop():
    if np.random.rand() < 0.1:
      print("Stopping from id: %d\n" % worker_id)
      coord.request_stop()
      pass
    else:
      print("Working on id: %d\n" % worker_id)
      pass
    time.sleep(1)

coord = tf.train.Coordinator()
threads = [
    threading.Thread(target=myloop, args=(coord, i, )) for i in range(5)]

for t in threads:
  t.start()

coord.join(threads)
