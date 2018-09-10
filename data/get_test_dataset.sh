#!/bin/bash

URL=http://merlin.fit.vutbr.cz/elevation/geoPose3K_final_publish.tar.gz
TAR_FILE=./data/geoPose3K_final_publish.tar.gz
wget $URL -O $TAR_FILE
tar -xvzf $TAR_FILE -C ./data/