#!/bin/bash

# please remember to give permission before use. "chmod +x install.sh"
# if installing to local python directory, may require "sudo bash install.sh"

function pip_install {
  for p in $@; do
    pip install $p
    if [ $? -ne 0 ]; then
      echo "could not install $p - abort, please install by hand."
      exit 1
    fi
  done
}

pip_install scipy
pip_install sklearn
pip_install configobj
