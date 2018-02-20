import json
import time
import gzip
import tarfile

from datetime import datetime
from multiprocessing import Pool, cpu_count

import copy
import requests

class JobAPI:
  def __init__(self, sleep_time = 0.5):
    self.sleep_time = sleep_time

    self.submitted_jobs = []
    self.completed_jobs = []
    self.api_url = 'http://localhost:5000/api/v1'
    self.json_header = { 'Content-Type' : 'application/json' }

  def __asyncCallback(self, result):
    return result

  def __asyncErrorCallback(self, error):
    print('Error: %s' % error)

  def __startJob(self, job):
    task = requests.put(self.api_url + ('/tasks/%(id)d' % job['api_data']), headers = self.json_header, data = json.dumps({'started': datetime.now()}))

    return task

  def __updateJobStatus(self, job, result, dt):
    left = result._number_left * result._chunksize

    if left == job['api_data']['left']:
      return

    api_data = {
      'left': left,
      'elapsed_time': dt
    }

    task = requests.put(self.api_url + ('/tasks/%(id)d' % job['api_data']), headers = self.json_header, data = json.dumps(api_data))

    job['api_data'].update(task.json())

    return job['api_data']['status_id']

  def __saveOutput(self, job_data, result, elapsed_time, status_id):
    args_data = [list(arg) for arg in job_data['args']]

    new_api_data = {
      'elapsed_time' : elapsed_time,
      'status_id': status_id,
      'completed': datetime.now().timestamp()
    }

    job_data['api_data'].update(new_api_data)

    data = {
      'api_data': job_data['api_data'],
      'args' : args_data,
      'result' : result
    }

    data_str = json.dumps(data, sort_keys = True)
    filename = 'bin/data/%(name)s_%(id)d_%(completed).0f.json.gz' % job_data['api_data']

    with gzip.open(filename, 'wb') as fp:
      fp.write(data_str.encode())

    print("Saved to %s." % filename)

    new_api_data['output_file'] = filename

    task = requests.put(self.api_url + ('/tasks/%(id)d' % data['api_data']), headers = self.json_header, data = json.dumps(new_api_data))

    return task

  def loadData(filename):
    with gzip.open(filename, 'rb') as fp:
      return json.loads(fp.read().decode())

  def submit(self, func, N, *args, p = None, bs = 16, desc = ''):
    if p == None:
      p = cpu_count()

    job_data = {
      'func': func,
      'args': args,
      'api_data': {
        'name': func.__name__,
        'description': desc,
        'cpu_count': p,
        'size': N,
        'block_size': bs
      }
    }

    task = requests.post(self.api_url + '/tasks', headers = self.json_header, data = json.dumps(job_data['api_data']))

    job_data['api_data'].update(task.json())

    self.submitted_jobs.append(job_data)
    return job_data

  def process(self):
    for job in self.submitted_jobs:
      try:
        job_data = job['api_data']

        print('Starting "%(name)s" with %(cpu_count)d processors and block size %(block_size)d.' % job_data)

        args_cpy = copy.deepcopy(job['args'])
        args_map = map(tuple, zip(*job['args']))

        task = requests.put(self.api_url + ('/tasks/%(id)d' % job_data), headers = self.json_header, data = json.dumps({'status_id': 3, 'started': datetime.now().timestamp()}))

        job_data.update(task.json())
        job_status = job_data['status_id']

        t0 = time.time()
        with Pool(job_data['cpu_count']) as workers:
          result = workers.starmap_async(job['func'], args_map, job_data['block_size'], callback = self.__asyncCallback, error_callback = self.__asyncErrorCallback)

          msg = ''
          completed = False
          try:
            while result._number_left > 0:
              job_status = self.__updateJobStatus(job, result, time.time() - t0)
              if job_status == 6: # status == stopped
                raise KeyboardInterrupt
              time.sleep(self.sleep_time)

            self.__updateJobStatus(job, result, time.time() - t0)
            result_data = result.get()
            completed = True
            job_status = 2  # -> status_id for completed
          except KeyboardInterrupt:
            workers.terminate()
            job_status = 6  # -> status_id for stopped
            result_data = [x if x else float('nan') for x in result._value]
          except:
            raise

        dt = time.time() - t0

        print('Finishing "%s": N = %d, t*p/N = %.2f ms, t = %.2f s.' % (job_data['name'], job_data['size'], job_data['cpu_count'] * dt * 1e3 / job_data['size'], dt))

        self.__saveOutput(job, result_data, dt, job_status)

        self.completed_jobs.append(job)

        del task
      except KeyboardInterrupt:
        continue
      except:
        raise

