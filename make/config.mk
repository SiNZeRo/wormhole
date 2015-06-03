#-----------------------------------------------------
#  wormhole: the configuration compile script
#
#  This is the default configuration setup for all dmlc projects
#  If you want to change configuration, do the following steps:
#
#  - copy this file to the root of wormhole folder
#  - modify the configuration you want
#  - type make or make -j n on each of the folder
#----------------------------------------------------

# choice of compiler
export CC = gcc
export CXX = g++
export MPICXX = mpicxx

# whether use HDFS support during compile
USE_HDFS = 0

# whether use AWS S3 support during compile
USE_S3 = 0

# path to libjvm.so
LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server
