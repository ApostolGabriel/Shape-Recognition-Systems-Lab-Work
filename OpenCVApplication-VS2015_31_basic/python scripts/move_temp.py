from os import walk
from pathlib import Path

filenames = next(walk('../Project data/original/'), (None, None, []))[2]  # [] if no file

print(filenames)

first_letters_class = filenames[0][ 0 : 4 ]
for filename in filenames:
  first_letters = filename[ 0 : 4 ]
  if first_letters == first_letters_class:
    last_letters = filename[ len(filename) - 6 : len(filename) ]
    if last_letters == '-1.gif' or last_letters == '-2.gif' or last_letters == '01.gif' or last_letters == '02.gif':
      img_path_source = '../Project data/original/' + filename
      img_path_dest = '../Project data/templates/' + filename
      try:
        Path(img_path_source).rename(img_path_dest)
      except:
        continue
  else:
    first_letters_class = first_letters
