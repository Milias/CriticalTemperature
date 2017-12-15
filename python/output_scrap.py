with open('out.txt', 'r') as f:
  file_str = f.read()

started = []
finished = []
for line in file_str.split('\n'):
  if 'Starting' in line:
    started.append(line[line.find(' ') + 1:])
  elif 'Finished' in line:
    finished.append(line[line.find(' ') + 1:])

print('\n'.join(sorted(list(set(started) - set(finished)))))

