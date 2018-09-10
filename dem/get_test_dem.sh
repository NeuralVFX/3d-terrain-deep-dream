#!/bin/bash

URL=https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/IMG/USGS_NED_one_meter_x34y441_CO_Central_Western_2016_IMG_2018.zip
ZIP_FILE=./dem/x34y441_CO.zip
wget $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./dem/


