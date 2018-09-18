import os
import json
import gzip
import tarfile

import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import copy
import pickle
import base64
import hashlib
from hmac import compare_digest

import requests

class JobAPI_Item:
  save_folder = 'bin/data'

  def update_data(self, new_data):
    self.data['api_data'].update(new_data)

  def __getitem__(self, key):
    return self.data['api_data'][key]

  def __getattr__(self, attr):
    return self.data.get(attr)

class JobAPI_Batch(JobAPI_Item):
  def __init__(self, api_data = {}):
    self.data = {'api_data': api_data}

  def add_file(self, new_file, path = '/tmp/'):
    # Here we add the file created for the job to the
    # tar.gz container associated with the batch.
    #
    # If the tar.gz file exists, it is first read, its
    # contents extracted to /tmp/old (by default)
    # and then compressed again with the new file.

    tar_fp = '%s/%s' % (self.save_folder, self.data['api_data']['output_file'])
    files_to_add = []

    if os.path.exists(tar_fp):
      with tarfile.open(tar_fp, 'r') as fp:
        files_to_add.extend([path + 'old/' + file for file in fp.getnames()])
        fp.extractall(path = path + 'old')

    with tarfile.open(tar_fp, 'w:gz') as fp:
      for file in files_to_add:
        if os.path.basename(file) == os.path.basename(new_file):
          continue

        fp.add(file, arcname = os.path.basename(file))
        os.remove(file)

      fp.add(new_file, arcname = os.path.basename(new_file))
      os.remove(new_file)

    return tar_fp

  def load_files(self, path = '/tmp/'):
    # Extracts all files stored in the 'output_file'
    # associated with this batch, if it exists, and
    # returns a list of filenames.
    #
    # By default it extracts to /tmp.

    tar_fp = '%s/%s' % (self.save_folder, self.data['api_data']['output_file'])

    files = []
    if os.path.exists(tar_fp):
      with tarfile.open(tar_fp, 'r') as fp:
        files.extend([path + name for name in fp.getnames()])
        fp.extractall(path = path)

    return files

class JobAPI_Job(JobAPI_Item):
  def __init__(self, batch, api_data, func, *args):
    self.__batch = batch

    self.data = {'api_data': api_data}

    self.func = func
    self.args = args

  def update_saved_data(self, new_data, path = '/tmp/'):
    # Here the data associated with this job
    # is saved or updated, with data being a dictionary
    # containing the new data.

    self.data.update(new_data)

    data_str = json.dumps(self.data, sort_keys = True)
    filename = '%(name)s_%(id)d.json' % self
    abs_filename = path + filename

    # First we save the json file to /tmp
    with open(abs_filename, 'w+') as fp:
      fp.write(data_str)

    # Then it is added to the batch container
    self.__batch.add_file(abs_filename, path = path)

class JobAPI_Iterator:
  def __init__(self, job_api):
    self.job_api = job_api

  def __iter__(self):
    return self

  def __next__(self):
    if self.job_api.process_index == len(self.job_api.jobs):
      raise StopIteration

    job = self.job_api.jobs[self.job_api.process_index]
    self.job_api.process_index += 1

    return job

class JobAPI:
  def __init__(self, api_token):
    self.__sleep_time = 0.1

    self.__api_url = 'http://localhost:5000/api/v1'
    self.__api_header = { 'Content-Type' : 'application/json' }
    self.__api_auth = (api_token, 'api_token')

    self.__session = requests.Session()
    self.__session.auth = self.__api_auth
    self.__session.headers.update(self.__api_header)

    self.__hash_key = b'QkpQRUZiU0UzdFlZOUJ6NzVKWjR6akdrSzh3N3Y0ZEtBdmNna1R1ZWRDZ21lQmtXbUJXZTJ4bjJqd3U0M0IzRQ=='

    self.__state = {
      'batch': None,
      'active_job': None,
      'jobs': [],
      'loaded_jobs': [],
      'process_index': 0,
      'processing': False,
    }

  def __getattr__(self, attr):
    return self.__state.get(attr)

  def __hasher(self):
    return hashlib.blake2b(key = base64.b64decode(self.__hash_key))

  def __api_call(self, url, call_type = 'get', *args, **kwargs):
    return getattr(self.__session, call_type)(self.__api_url + url, *args, **kwargs)

  def __update_batch(self):
    if self.__state['batch']:
      response = self.__api_call('/batches/%d' % self.__state['batch']['id'])

      if response == 200:
        self.__state['batch'].update_data(response.json())
      else:
        raise RuntimeError('Error %d: %s' % (response.status_code, response.json()))

  def __start_job(self, job):
    response = self.__api_call('/jobs/%d' % job['id'], 'put', data = json.dumps({'started': datetime.now().timestamp()}))

    if response.status_code == 200:
      job.update_data(response.json())
    else:
      raise RuntimeError('Error %d: %s' % (response.status_code, response.json()))

  def __update_job(self, job, new_data):
    # Updates API-related data for the given job.
    #
    # If save_data == True, it will also update the
    # saved file.

    response = self.__api_call('/jobs/%d' % job['id'], 'put', data = json.dumps(new_data))

    if response.status_code == 200:
      job.update_data(response.json())

    else:
      raise RuntimeError('Error %d: %s' % (response.status_code, response.json()))

  def __hash_args(self, func, *args):
    args_cpy = copy.deepcopy(args)
    func_cpy = copy.copy(func)

    args_bytes = base64.b64encode(pickle.dumps((func_cpy, args_cpy)))

    hasher = self.__hasher()
    hasher.update(args_bytes)
    args_hash = hasher.hexdigest()

    return {'func': func_cpy, 'args': args_cpy, 'bytes': args_bytes, 'hash': args_hash}

  def __process_job(self, job):
    print('Starting "%(name)s" with %(cpu_count)d processors and block size %(block_size)d.' % job)
    self.__state['active_job'] = job

    args_cpy = copy.deepcopy(job.args)
    args_map = map(tuple, zip(*args_cpy))

    # Set status_id to 3 -> active
    self.__update_job(job, {'status_id': 3})
    job_status = job['status_id']

    t0 = time.time()
    with Pool(job['cpu_count']) as workers:
      result = workers.starmap_async(job.func, args_map, job['block_size'])

      try:
        while result._number_left > 0:
          if result._number_left * result._chunksize != job['left']:
            self.__update_job(job, {'elapsed': time.time() - t0, 'left': result._number_left * result._chunksize})

          if job['status_id'] == 6: # status == stopped
            raise KeyboardInterrupt

          time.sleep(self.__sleep_time)

        self.__update_job(job, {'elapsed': time.time() - t0, 'left': result._number_left * result._chunksize})
        result_data = result.get()
        job_status = 2 # -> status_id for completed

      except KeyboardInterrupt:
        workers.terminate()
        job_status = 6 # -> status_id for stopped
        result_data = [x if x else float('nan') for x in result._value]

    dt = time.time() - t0

    self.__update_job(job, {
        'elapsed': dt,
        'completed': datetime.now().timestamp(),
        'status_id': job_status
      }
    )

    job.update_saved_data({'result': result_data})

    print('Finishing "%s": N = %d, t*p/N = %.2f ms, t = %.2f s.' % (job['name'], job['size'], job['cpu_count'] * dt * 1e3 / job['size'], dt))

  def __load_batch_jobs(self, batch):
    files = batch.load_files()

    for file in files:
      with open(file, 'r') as fp:
        data = json.loads(fp.read())

        hasher = self.__hasher()
        hasher.update(data['args_data'].encode())
        args_hash_file = hasher.hexdigest()

        response = self.__api_call('/jobs/%d' % data['api_data']['id'])

        if response.status_code == 200:
          job_api_data = response.json()
        else:
          raise RuntimeError('Error %d: %s' % (response.status_code, response.json()))

        args_hash_db = job_api_data['args_hash']

        if compare_digest(args_hash_file, args_hash_db):
          func, args = pickle.loads(base64.b64decode(data['args_data'].encode()))
        else:
          raise RuntimeError('>>> HASH CHECK FAILED <<<')

        job = JobAPI_Job(batch, job_api_data, func, *args)
        job.data['result'] = data['result']

        self.__state['loaded_jobs'].append(job)

  def new_batch(self, name, description):
    api_data = {
      'name': name,
      'description': description,
    }

    response = self.__api_call('/me/batches', 'post', data = json.dumps(api_data))

    if response.status_code == 201:
      batch = JobAPI_Batch(response.json())
      self.__state['batch'] = batch

      return batch
    else:
      raise RuntimeError('Error %d: %s' % (response.status_code, response.json()))

  def load_batch(self, batch_id = None):
    if batch_id:
      response = self.__api_call('/batches/%d' % batch_id)
    else:
      response = self.__api_call('/me/batches/last')

    if response.status_code == 200:
      batch = JobAPI_Batch(response.json())
      self.__state['batch'] = batch
      self.__load_batch_jobs(batch)

    else:
      raise RuntimeError('Error %d: %s' % (response.status_code, response.json()))

    return batch

  def submit(self, func, *args, **kwargs):
    if not self.__state['batch']:
      raise RuntimeError('No batch selected.')

    hash_data = self.__hash_args(func, *args)

    api_data = {
      'name': func.__name__,
      'batch_id': self.__state['batch']['id'],
      'status_id': 1,
      'cpu_count': cpu_count(),
      'args_hash': hash_data['hash']
    }

    api_data.update(kwargs)
    response = self.__api_call('/me/jobs', 'post', data = json.dumps(api_data))

    if response.status_code == 201:
      job = JobAPI_Job(self.__state['batch'], response.json(), func, *args)
      self.__state['jobs'].append(job)

      job.update_saved_data({
          'args_data': hash_data['bytes'].decode()
        }
      )

      return job
    else:
      raise RuntimeError('Error %d: %s' % (response.status_code, response.json()))

  def process(self):
    for job in JobAPI_Iterator(self):
      try:
        self.__process_job(job)
      except Exception as exc:
        self.__update_job({'status_id': 5})
        raise exc

