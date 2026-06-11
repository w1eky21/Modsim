# convenience variables that must exist
prefix=/home/stijn/local
exec_prefix=${prefix}

export PATH="$exec_prefix/bin":$PATH
export PYTHONPATH="/home/stijn/local/lib/python3.13/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="${exec_prefix}/lib:$LD_LIBRARY_PATH"
export LHAPDF_DATA_PATH="$prefix/share/LHAPDF/"

