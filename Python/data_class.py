#!/usr/bin/env python3

from dataclasses import dataclass, field

@dataclass
class MyData:
  """My data class"""

  # name field
  name: str

  # dob field
  dob: str

  # status field
  info: str = field(default='formalism')

def main():
  """Some function"""

  data = MyData('dima', '01/02/23')
  print(data.name, data.dob, data.info)
  print(data)

if __name__ == "__main__":

  main()
  
