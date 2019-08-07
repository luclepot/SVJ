#!/bin/bash
HEAD=$(git rev-parse --show-toplevel)
source "$HEAD/autoencode/setup.sh"
source "$HEAD/autoencode/build.sh"
