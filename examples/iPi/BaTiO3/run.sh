#!/bin/bash

# remove the driver file if it already exists
rm /tmp/ipi_BaTiO3
# make sure rascal can be imported if not installed
export PYTHONPATH="../../../build_b/:$PYTHONPATH"
# path to the i-Pi driver
RASCAL_DRIVER="../../../scripts/ipi_driver.py"
# i-Pi executable
IPI="i-pi"

# initialize the socket and set up the simulation
$IPI input.xml &
sleep 2
# send simulation
$RASCAL_DRIVER -u -a BaTiO3 -m BaTiO3_model.json -s BaTiO3_cubic_3x3x3.xyz &
wait
