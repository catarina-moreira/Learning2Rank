# Makefile for Python SVM-struct API, 03.07.04

export

CC = gcc
LD = gcc
#CC = attolcc -mempro -perfpro -block -proc -- gcc
#LD = attolcc -mempro -perfpro -block -proc -- gcc
CFLAGS = -O3 -fomit-frame-pointer -ffast-math -Wall 
LDFLAGS = -O3 -lm -Wall
#CFLAGS = -g -Wall
#LDFLAGS = -g -lm
#CFLAGS = -pg -Wall
#LDFLAGS = -pg -lm -Wall 

# Specify which Python to use!  Just leaving this "python" is fine if
# you want to use the library corresponding to the python interpreter
# you get when you just type python on the command line.
PYTHON = python


# Happily, thanks to the built-in distutils module, the Python
# interpreter is able to return many variables relating to its
# installation including where to look for the library, the library
# name, include files, and the special compiler directives that allow
# for dynamic loading of modules in programs that embed Python!
define PYTHON_COMMAND
-c "from distutils import sysconfig; print sysconfig.get_config_var('R')"
endef
PYTHON_LIB_DIR := $(shell $(PYTHON) $(subst R,LIBPL,$(PYTHON_COMMAND)))
PYTHON_LIB := $(basename $(patsubst lib%,%,$(shell $(PYTHON) $(subst R,LIBRARY,$(PYTHON_COMMAND)))))
PYTHON_INCLUDE := $(shell $(PYTHON) $(subst R,INCLUDEPY,$(PYTHON_COMMAND)))
PYTHON_SHARED := $(shell $(PYTHON) $(subst R,LINKFORSHARED,$(PYTHON_COMMAND)))
PYTHON_OTHER_LIBS := $(shell $(PYTHON) $(subst R,LIBS,$(PYTHON_COMMAND)))

PYTHON_LD_FLAGS := $(PYTHON_SHARED) -L$(PYTHON_LIB_DIR) -l$(PYTHON_LIB) $(PYTHON_OTHER_LIBS)
PYTHON_CC_FLAGS := -I$(PYTHON_INCLUDE)

# Check if the numarray package appears to be installed.
ifneq ($(wildcard $(PYTHON_INCLUDE)/numarray/),)
	EXTRA_DEFINES := $(EXTRA_DEFINES) -D NUMARRAY
else
# Check if the Numeric package appears to be installed.
ifneq ($(wildcard $(PYTHON_INCLUDE)/Numeric/),)
	EXTRA_DEFINES := $(EXTRA_DEFINES) -D NUMERIC
endif
endif

#EXTRA_DEFINES :=

all: svm_map_learn_hideo svm_map_classify

help:
	python -c 'import svmstruct;help(svmstruct)'

.PHONY: clean
clean: svm_light_clean svm_struct_clean
	rm -f *.o *.tcov *.d core gmon.out *.stackdump svm_map_* *.pyc
cleanest: clean
	find . -name "*~" -exec rm {} \;
	rm -f model.*

#-----------------------#
#----   SVM-light   ----#
#-----------------------#
svm_light_hideo_noexe: 
	cd svm_light; make svm_learn_hideo_noexe

svm_light_loqo_noexe: 
	cd svm_light; make svm_learn_loqo_noexe

svm_light_clean: 
	cd svm_light; make clean

#----------------------#
#----  STRUCT SVM  ----#
#----------------------#

svm_struct_noexe: 
	cd svm_struct; make svm_struct_noexe

svm_struct_clean: 
	cd svm_struct; make clean


#--------------------------#
#----  SVM python API  ----#
#--------------------------#

python-link:
	MACOSX_DEPLOYMENT_TARGET=10.3
	export MACOSX_DEPLOYMENT_TARGET
	python link-script.py

svm_map_classify: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_classify.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o 
	$(LD) $(LDFLAGS) svm_struct_api.o svm_struct/svm_struct_classify.o svm_light/svm_common.o svm_struct/svm_struct_common.o -o svm_map_classify $(PYTHON_LD_FLAGS) $(LIBS)

svm_map_learn_loqo: svm_light_loqo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_learn.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o 
	$(LD) $(LDFLAGS) svm_light/svm_loqo.o svm_light/pr_loqo/pr_loqo.o svm_light/svm_learn.o svm_light/svm_common.o $(filter %.o,$^) -o svm_map_learn $(PYTHON_LD_FLAGS) $(LIBS)

svm_map_learn_hideo: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o  svm_struct/svm_struct_learn.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o
	$(LD) $(LDFLAGS) svm_light/svm_hideo.o svm_light/svm_learn.o svm_light/svm_common.o $(filter %.o,$^) -o svm_map_learn $(PYTHON_LD_FLAGS) $(LIBS)


svm_struct_api.o: svm_struct_api.c svm_struct_api.h svm_struct_api_types.h svm_struct/svm_struct_common.h
	$(CC) -c $(CFLAGS) $(PYTHON_CC_FLAGS) $(EXTRA_DEFINES) $(filter %.c,$^) -o svm_struct_api.o

