/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/


/* If the Numeric module has been loaded, that means SVM^python
   doesn't have to copy over the model arrays, saving time and
   memory.  It also makes the Scipy module easier to use! */

#include <Python.h>
#ifdef NUMARRAY
#include <numarray/libnumarray.h>
#include <numarray/arrayobject.h>
#elif defined NUMERIC
#include <Numeric/arrayobject.h>
#endif /* NUMERIC and NUMARRAY */
#include <stdio.h>
#include <string.h>
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"

/* This is the module that the user functions and other declarations
   come from. */
PyObject *pModule;
/* In the C code, the array sm.w has meaningful values starting at
   index 1 (so index 0 was ignored).  Further, in SVM^light's sparse
   WORD arrays, the first meaningful .wnum index was 1.  I found this
   very annoying for my applications, as my data structures were just
   based on regular arrays indexed from 0 so I was constantly adding
   (or forgetting to add) 1 here and there.

   The default is to have pStartingIndex=1.  In this case everything
   is considered to be indexed from 1, and the Python glue code does
   the translation between the two transparently.  The Python code
   behaves like the C code (that is, sm.w[0] is useless, and the 0
   entry of the vector returned by psi is invalid).  If you want
   things to index from 0, define in the PYTHON_PARAM dictionary a
   string 'index_from_one' that maps to False.  If this variable is
   not defined or evaluates to True, then the default behavior is
   assumed. */
int pStartingIndex;
#define PYTHON_PARAM			"svmpython_parameters"
#define PYTHON_PARAM_INDEX_FROM_ONE	"index_from_one"

/* This is the default module name in case the --m <module> option is
   not specified on the command line. */
#define PYTHON_MODULE_NAME		"svmstruct_map"

/* The following functions are required.  The program will exit in the
   event that they are not implemented.  It's worth noting, however,
   that if a function is not called by the program, its absence will
   not be noticed, e.g., if init_model (only required during learning)
   is absent during classification, the program will not care. */
#define PYTHON_READ_EXAMPLES		"read_struct_examples"
#define PYTHON_INIT_MODEL		"init_struct_model"
#define PYTHON_CLASSIFY_EXAMPLE		"classify_struct_example"
#define PYTHON_PSI			"psi"
/* The following functions are technically not required because some
   default behavior is defined, but you probably will want to
   implement them anyway since the default behavior will not be
   generally acceptable.  If they are not implemented a warning
   message will be output. */
#define PYTHON_FIND_MOST_VIOLATED	"find_most_violated_constraint"
#define PYTHON_LOSS			"loss"
#define PYTHON_PARSE_ARGUMENTS		"parse_struct_parameters"
/* The following functions are not required, and if not present in the
   python module some default behavior will be performed.  Unlike the
   above unrequired functions, it is quite reasonable to rely on the
   default behavior of these functions in many cases, and so if not
   implemented a warning message will not be printed out. */
#define PYTHON_INIT_CONSTRAINTS		"init_struct_constraints"
#define PYTHON_PRINT_LEARNING_STATS	"print_struct_learning_stats"
#define PYTHON_PRINT_TESTING_STATS	"print_struct_testing_stats"
#define PYTHON_EVAL_PREDICTION		"eval_prediction"
#define PYTHON_WRITE_MODEL		"write_struct_model"
#define PYTHON_READ_MODEL		"read_struct_model"
#define PYTHON_WRITE_LABEL		"write_label"
#define PYTHON_PRINT_HELP		"print_struct_help"

#define PYTHON_CALL(RET,FUN,ARG) /*if (PyErr_Occurred()) { PyErr_Print(); }*/ PyErr_Clear(); RET = PyObject_CallObject(FUN,ARG); if (PyErr_Occurred()) { PyErr_Print(); Py_Exit(1); }

/************* PYTHON SVM MODULE DEFINITION *********/

static PyObject* emb_classify_example(PyObject *self, PyObject *args);
static PyObject* emb_create_svector(PyObject *self, PyObject *args);
static PyObject* emb_create_doc(PyObject *self, PyObject *args);
static PyObject* emb_kernel(PyObject *self, PyObject *args);

static PyMethodDef EmbMethods[] = {
  {"classify_example", emb_classify_example, 2,
   "Classify a feature vector with the model's kernel function."},
  {"create_doc", emb_create_doc, METH_VARARGS,
   "Create a Python document object."},
  {"create_svector", emb_create_svector, METH_VARARGS,
   "Create a Python support vector object."},
  {"kernel", emb_kernel, 3,
   "Evaluate a kernel function on two feature vectors."},
  {NULL, NULL, 0, NULL}
};

/************* PYTHON EMBEDDED INITIALIZATION/FINALIZATION *********/

void        api_initialize(char * name) {
  /* This is called before anything else in the API, allowing whatever
     initialization is required. */
  Py_SetProgramName(name);
  Py_Initialize();

#ifdef NUMARRAY
  import_libnumarray();
  import_libnumeric();
#elif defined NUMERIC
  import_array();
#endif

  Py_InitModule("svmlight", EmbMethods);
  /* Create the blank object type. */
  if (PyRun_SimpleString("class SVMBlank(object):\n\tpass\n")) {
    fprintf(stderr, "Could not define SVMBlank type!\n");
    Py_Exit(1);
  }
}

void	    api_load_module(const char *module_name) {
  /* This is typically called soon-ish after api_initialize, so the
     module with the user functions can be imported.  Only the first
     call to this function is effective.  If module_name==NULL, a
     default name is assumed. */
  PyObject *pDict, *pValue, *pName, *pParam;
  static int alreadyCalled = 0;
  if (alreadyCalled) return;
  alreadyCalled = 1;
  if (module_name==NULL) module_name = PYTHON_MODULE_NAME;
  
  /* Load the module! */
  pName = PyString_FromString(module_name);
  pModule = PyImport_Import(pName);
  Py_DECREF(pName);
  if (pModule == NULL) {
    fprintf(stderr, "Could not load module %s!\n", module_name);
    fprintf(stderr, "Is your PYTHONPATH environment variable set properly?\n");
    Py_Exit(1);
  } else {
    //printf("Loaded module %s!\n", module_name);
  }
  /* Detect parameters. */
  pDict = PyModule_GetDict(pModule);
  pValue = pParam = NULL;
  pStartingIndex = 1;
  if (!PyMapping_HasKeyString(pDict,PYTHON_PARAM)) return;
  pParam = PyMapping_GetItemString(pDict,PYTHON_PARAM);

  /* Check if we should index from one. */
  if (PyMapping_HasKeyString(pParam, PYTHON_PARAM_INDEX_FROM_ONE)) {
    pValue = PyMapping_GetItemString(pParam,PYTHON_PARAM_INDEX_FROM_ONE);
    pStartingIndex = (pValue && PyObject_IsTrue(pValue)) ? 1 : 0;
    Py_XDECREF(pValue);
  }
  Py_XDECREF(pParam);
}

void        api_finalize() {
  /* This is called after everything else in the API, allowing
     whatever cleanup is required. */
  if (PyErr_Occurred()) { PyErr_Print(); Py_Exit(1); }
  Py_Finalize();
}

/************ PYTHON EMBEDDED CONVENIENCE FUNCTIONS **********/

inline int pythonSetI(PyObject *obj, char*attr_name, long i) {
  PyObject *integer = PyInt_FromLong(i);
  int retval = PyObject_SetAttrString(obj, attr_name, integer);
  Py_DECREF(integer);
  return retval;
}
inline int pythonSetF(PyObject *obj, char*attr_name, double d) {
  PyObject *number = PyFloat_FromDouble(d);
  int retval = PyObject_SetAttrString(obj, attr_name, number);
  Py_DECREF(number);
  return retval;
}
inline int pythonSetS(PyObject *obj, char*attr_name, char *s) {
  PyObject *string = PyString_FromString(s);
  int retval = PyObject_SetAttrString(obj, attr_name, string);
  Py_DECREF(string);
  return retval;
}
inline int pythonGetI(PyObject *obj, char*attr_name, long *i) {
  PyObject *value = PyObject_GetAttrString(obj, attr_name);
  if (!value || !PyNumber_Check(value)) return 0;
  *i = PyInt_AsLong(value);
  Py_DECREF(value);
  return 1;
}
inline int pythonGetF(PyObject *obj, char*attr_name, double *d) {
  PyObject *value = PyObject_GetAttrString(obj, attr_name);
  if (!value || !PyNumber_Check(value)) return 0;
  *d = PyFloat_AsDouble(value);
  Py_DECREF(value);
  return 1;
}
inline int pythonGetS(PyObject *obj, char*attr_name, char *s, int maxlen) {
  char *str;
  PyObject *value2 = PyObject_GetAttrString(obj, attr_name), *value;
  if (!value2) return 0; // No check...Anything can be represented as a string.
  value = PyObject_Str(value2);
  Py_DECREF(value2);
  str = PyString_AsString(value);
  if (maxlen<0)
    strcpy(s, str); // Unsafe, but convenient!  Yay!
  else
    strncpy(s, str, maxlen);
  Py_DECREF(value);
  return 1;
}


/* Acts like SetItemString but with reference stealing for a shortcut! */
inline int pythonMap(PyObject*mapping, char*key, PyObject*value) {
  if (PyMapping_HasKeyString(mapping, key))
    PyMapping_DelItemString(mapping, key);
  int result = PyMapping_SetItemString(mapping, key, value);
  Py_DECREF(value);
  return result;
}

/* This creates a totally blank instance with absolutely no data. */
PyObject* pythonCreateBlank() {
  PyObject *obj;
  PyRun_SimpleString("_svmblank = SVMBlank()");
  obj = PyMapping_GetItemString(PyModule_GetDict(PyImport_AddModule
						 ("__main__")), "_svmblank");
  if (obj != NULL) {
    //PyObject_Print(obj, stdout, 0); printf("\n");
  } else {
    fprintf(stderr, "Could not create blank object\n");
    Py_Exit(1);
  }
  /* Without the following, the reference would stick around until we
     created another blank item.  That isn't so terrible, but why
     wait? */
  PyMapping_DelItemString(PyModule_GetDict(PyImport_AddModule
					   ("__main__")), "_svmblank");
  return obj;
}

PyObject* sampleToPythonObject(SAMPLE sample) {
  PyObject *pExamples;
  int i;
  /* Build the example list. */
  pExamples = PyList_New(sample.n);
  for (i=0; i<sample.n; ++i) {
    PyList_SET_ITEM(pExamples, i, Py_BuildValue
		    ("(OO)", sample.examples[i].x.py_pattern,
		     sample.examples[i].y.py_label));
  }
  return pExamples;
}

/* Given a support vector, return a python representation of the
   support vector.  Consider the returned object as a sequence which
   we shall call sv.  In the interpreter, len(sv) would equal the
   number of features in the support vector.  sv[i][0] would equal the
   feature number, and sv[i][1] the feature value. */
PyObject* svToPythonObject(SVECTOR *sv) {
  PyObject *py_words, *py_sv, *py_list;
  int num_features = 0, num_svs = 0;
  SVECTOR *temp_sv;
  /* Create the empty list. */
  for (temp_sv = sv; temp_sv; temp_sv = temp_sv->next) num_svs++;
  py_list = PyTuple_New(num_svs);
  num_svs = 0;
  //printf("Adding stuff.\n");
  while (sv) {
    int i=0;
    WORD *temp = sv->words;
    //printf("Adding more stuff.\n");
    /* Create the empty object. */
    py_sv = pythonCreateBlank();
    //Py_DECREF(py_sv);
    //printf("%d\n", py_sv->ob_refcnt);
    /* Count the number of features. */
    num_features = 0;
    while ((temp++)->wnum) ++num_features;
    /* Create the tuple. */
    py_words = PyTuple_New(num_features);
    for (temp=sv->words; temp->wnum; ++temp) {
      PyObject *temp_tuple = PyTuple_New(2);
      PyTuple_SET_ITEM(temp_tuple, 0, PyInt_FromLong
		       (temp->wnum-1+pStartingIndex));
      PyTuple_SET_ITEM(temp_tuple, 1, PyFloat_FromDouble(temp->weight));
      PyTuple_SET_ITEM(py_words,i++,temp_tuple);
    }
    PyObject_SetAttrString(py_sv, "words", py_words);
    Py_DECREF(py_words);
    /* Copy over other information. */
    pythonSetI(py_sv, "kernel_id", sv->kernel_id);
    pythonSetF(py_sv, "factor", sv->factor);
    pythonSetS(py_sv, "userdefined", sv->userdefined);
    /* Add it to the list and move on. */
    PyTuple_SET_ITEM(py_list, num_svs++, py_sv);
    sv = sv->next;
  }
  /* Return the list. */
  return py_list;
}

SVECTOR* pythonObjectToSingleSV(PyObject *py_sv) {
  SVECTOR *sv;
  PyObject *pTemp, *py_words;
  WORD *words;
  char isNumbers=0;
  int n, i;
  /* Stuff for the thing. */
  char *userdefined = "";
  double factor = 1.0;
  long kernel_id = 0;

  if (!PyObject_HasAttrString(py_sv, "words")) {
    /* Well, we should at LEAST have that.  Jeez... */
    return NULL;
  }
  py_words = PyObject_GetAttrString(py_sv, "words");
  /* Check whether this is a list of sequences or a list of numbers. */
  if ((n=PySequence_Size(py_words)) > 0) {
    pTemp=PySequence_GetItem(py_words, 0);
    isNumbers = PyNumber_Check(pTemp);
    Py_DECREF(pTemp);
  }
  words = (WORD*)my_malloc((n+1)*(sizeof(WORD)));
  words[0].wnum = 0;
  if (isNumbers) {
    /* This is a list of numbers! */
    /*for (i=pStartingIndex; i<n; ++i) {
      words[i].wnum = i+1-pStartingIndex;
      pTemp = PySequence_GetItem(py_words, i);
      words[i].weight = PyFloat_AsDouble(pTemp);
      Py_DECREF(pTemp);
      }*/
    for (i=0; i<n-pStartingIndex; ++i) {
      words[i].wnum = i+1;
      pTemp = PySequence_GetItem(py_words, i+pStartingIndex);
      words[i].weight = PyFloat_AsDouble(pTemp);
      Py_DECREF(pTemp);
    }
  } else {
    PyObject *pFirst, *pSecond;
    /* This is a list of sequences, presumably! */
    for (i=0; i<n; ++i) {
      pTemp=PySequence_GetItem(py_words, i);
      pFirst=PySequence_GetItem(pTemp, 0);
      pSecond=PySequence_GetItem(pTemp, 1);
      if (pSecond && pFirst) {
	words[i].wnum = PyInt_AsLong(pFirst)+1-pStartingIndex;
	words[i].weight = PyFloat_AsDouble(pSecond);
	Py_DECREF(pFirst);
	Py_DECREF(pSecond);
      } else {
	Py_XDECREF(pFirst);
	Py_XDECREF(pSecond);
	return NULL;
      }
      Py_DECREF(pTemp);
    }
  }
  words[i].wnum = words[i].weight = 0;
  Py_DECREF(py_words);
  /* Other attributes are easier, happily.  Get the factor, kernel_id,
     and userdefined. */
  pythonGetF(py_sv, "factor", &factor);
  pythonGetI(py_sv, "kernel_id", &kernel_id);
  if (PyObject_HasAttrString(py_sv, "userdefined")) {
    pTemp = PyObject_GetAttrString(py_sv, "userdefined");
    if (!pTemp || !PyString_Check(pTemp)) {
      free(words);
      Py_XDECREF(pTemp);
      return NULL;
    }
    userdefined = PyString_AsString(pTemp);
  }
  /* Create the S-vector. */
  sv = create_svector(words, userdefined, factor);
  free(words);
  Py_DECREF(pTemp);
  sv->kernel_id = kernel_id;
  return sv;
  
}

/* Given a Python object supposedly containing an support vector,
   return an SVECTOR, or NULL if an error occurred. */
SVECTOR* pythonObjectToSV(PyObject *py_sv) {
  /* Create the Psi vector from the returned value. */
  if (PySequence_Check(py_sv)) {
    SVECTOR *sv, *temp_sv;
    PyObject *pTemp = NULL;
    int n, i;
    /* It's a list of SVs. */
    if ((n=PySequence_Size(py_sv))<1) {
      return NULL;
    }
    pTemp = PySequence_GetItem(py_sv, 0);
    sv = pythonObjectToSingleSV(pTemp);
    Py_DECREF(pTemp);
    if (sv == NULL) return NULL;
    temp_sv = sv;
    sv->next = NULL;
    for (i=1; i<n; ++i) {
      pTemp = PySequence_GetItem(py_sv, i);
      temp_sv->next = pythonObjectToSingleSV(pTemp);
      if (temp_sv->next == NULL) {
	free_svector(sv);
	return NULL;
      }
      Py_DECREF(pTemp);
      temp_sv = temp_sv->next;
      temp_sv->next = NULL;
    }
    return sv;
  } else {
    return pythonObjectToSingleSV(py_sv);
  }
}

/* Given a document, create an equivalent Python object. */
PyObject* docToPythonObject(DOC *doc) {
  PyObject *py_doc, *py_svec;
  py_doc = pythonCreateBlank();
  /* Add the feature vector. */
  py_svec = svToPythonObject(doc->fvec);
  PyObject_SetAttrString(py_doc, "fvec", py_svec);
  Py_DECREF(py_svec);
  /* Add the other attributes. */
  pythonSetI(py_doc, "docnum", doc->docnum);
  pythonSetI(py_doc, "slackid", doc->slackid);
  pythonSetF(py_doc, "costfactor", doc->costfactor);
  return py_doc;
}

/* Given a Python object supposedly containing a document, return a
   newly allocated document, or NULL if an error occurred. */
DOC *pythonObjectToDoc(PyObject *py_doc) {
  DOC *doc;
  PyObject *py_svec;
  /* Variables from the document. */
  SVECTOR *svec;
  long docnum, slackid;
  double costfactor;

  if (!PyObject_HasAttrString(py_doc, "fvec")) {
    /* We should at least have the feature vector. */
    return NULL;
  }
  py_svec = PyObject_GetAttrString(py_doc, "fvec");
  //PyObject_Print(py_svec, stdout, 0); printf("\n");
  if (!py_svec || !(svec = pythonObjectToSV(py_svec)) || !svec) {
    return NULL;
  }
  Py_DECREF(py_svec);
  /* Well, we have a support vector.  Try to get the rest. */
  if (!pythonGetI(py_doc, "docnum", &docnum)) docnum = 0xdeadbeef;
  if (!pythonGetI(py_doc, "slackid", &slackid)) slackid = 0xdeadbeef;
  if (!pythonGetF(py_doc, "costfactor", &costfactor)) costfactor = 1.0;
  /* Create the document. */
  doc = create_example(docnum, 0, slackid, costfactor, svec);
  return doc;
}

/* Given a sparm, this returns the python object associated with this
   sparm, or sets the python object associated with this sparm if none
   is not yet associated. */
PyObject* sparmToPythonObject(STRUCT_LEARN_PARM *sparm) {
  PyObject *py_sparm, *argv, *argd, *obj;
  int i;
  char *lastKeyString=NULL;

  py_sparm = (PyObject*)sparm->py_sparm;
  if (!py_sparm) {
    /* Initialize the struct learning parameter's python object. */
    sparm->py_sparm = py_sparm = pythonCreateBlank();
    /* Synchronize the sparm to the object. */
    pythonSetF(py_sparm, "epsilon", sparm->epsilon);
    pythonSetF(py_sparm, "new_const_retrain", sparm->newconstretrain);
    pythonSetF(py_sparm, "c", sparm->C);
    pythonSetI(py_sparm, "slack_norm", sparm->slack_norm);
    pythonSetI(py_sparm, "loss_type", sparm->loss_type);
    pythonSetI(py_sparm, "loss_function", sparm->loss_function);
    /* Don't forget the user's command line arguments! */
    argd = PyDict_New();
    argv = PyList_New(sparm->custom_argc);
    for (i=0; i<sparm->custom_argc; ++i) {
      obj = PyString_FromString(sparm->custom_argv[i]);
      if (sparm->custom_argv[i][0]=='-') {
	lastKeyString = sparm->custom_argv[i]+2;
      } else {
	assert(lastKeyString);
	PyMapping_SetItemString(argd, lastKeyString, obj);
      }
      PyList_SetItem(argv, i, obj);
    }
    PyObject_SetAttrString(py_sparm, "argv", argv);
    PyObject_SetAttrString(py_sparm, "argd", argd);
  } else {

  }
  Py_INCREF(py_sparm);
  return py_sparm;
}

PyObject* smToPythonObject(STRUCTMODEL *sm) {
  PyObject *py_sm, *array=NULL, *svectors, *cobject;
  int i, array_size[1];

  py_sm = (PyObject*)sm->py_sm;
  if (!py_sm) {
    /* Initialize the structure model's python object. */
    sm->py_sm = py_sm = pythonCreateBlank();
    /* Synchronize the sm's unchanging fields to the object. */
    pythonSetI(py_sm, "size_psi", sm->sizePsi);
  }
  /* We keep the py_sm object around in the sm structure for a while. */
  Py_INCREF(py_sm);
  if (!sm->dirty) return py_sm;
  sm->dirty = 0;
  /* Set the pointer back to the holding sm. */
  cobject = PyCObject_FromVoidPtr(sm, NULL);
  PyObject_SetAttrString(py_sm, "cobj", cobject);
  Py_DECREF(cobject);
  /* Set that sm.w array! */
  array_size[0] = sm->sizePsi+pStartingIndex;
  //printf("asswipe %ld\n", sm->sizePsi);
#if defined(NUMERIC)||defined(NUMARRAY)
  /* Represent it as a Numeric.array.  This avoids copying!! */
  array = PyArray_FromDimsAndData(1, array_size, PyArray_DOUBLE,
				  (char*)(sm->w+1-pStartingIndex));
#else
  /* Represent it as a tuple.  Unfortunately we have to copy sm.w... */
  array = PyTuple_New(array_size[0]);
  if (sm->sizePsi > 0) {
    for (i=1-pStartingIndex; i<=sm->sizePsi; ++i) {
      PyTuple_SET_ITEM(array, i-1+pStartingIndex,
		       PyFloat_FromDouble(sm->w ? sm->w[i] : 0.0));
    }
  }
#endif
  PyObject_SetAttrString(py_sm, "w", array);
  Py_DECREF(array);

  /* Set some more attributes. */
  if (!sm->svm_model) { 
    return py_sm;
  }
  /* Add the simple scalar attributes. */
  pythonSetI(py_sm, "sv_num",        sm->svm_model->sv_num);
  pythonSetI(py_sm, "at_upper_bound",sm->svm_model->at_upper_bound);
  pythonSetF(py_sm, "b",             sm->svm_model->b);
  pythonSetI(py_sm, "totwords",      sm->svm_model->totwords);
  pythonSetI(py_sm, "totdoc",        sm->svm_model->totdoc);
  pythonSetF(py_sm, "loo_error",     sm->svm_model->loo_error);
  pythonSetF(py_sm, "loo_recall",    sm->svm_model->loo_recall);
  pythonSetF(py_sm, "loo_precision", sm->svm_model->loo_precision);
  pythonSetF(py_sm, "xa_error",      sm->svm_model->xa_error);
  pythonSetF(py_sm, "xa_recall",     sm->svm_model->xa_recall);
  pythonSetF(py_sm, "xa_precision",  sm->svm_model->xa_precision);
  pythonSetF(py_sm, "maxdiff",       sm->svm_model->maxdiff);
  /* Add the simple C scalar attributes from the kernel parameter. */
  pythonSetI(py_sm, "kernel_type",   sm->svm_model->kernel_parm.kernel_type);
  pythonSetI(py_sm, "poly_degree",   sm->svm_model->kernel_parm.poly_degree);
  pythonSetF(py_sm, "rbf_gamma",     sm->svm_model->kernel_parm.rbf_gamma);
  pythonSetF(py_sm, "coef_lin",      sm->svm_model->kernel_parm.rbf_gamma);
  pythonSetF(py_sm, "coef_const",    sm->svm_model->kernel_parm.coef_const);
  pythonSetS(py_sm, "custom",        sm->svm_model->kernel_parm.custom);
  /* Add the alpha array. */
  array_size[0] = sm->svm_model->sv_num; //+1;  //totdoc+2;
#if defined (NUMERIC)||defined(NUMARRAY)
  array = PyArray_FromDimsAndData(1, array_size, PyArray_DOUBLE,
				  (char*)(sm->svm_model->alpha));
#else
  array = PyTuple_New(array_size[0]);
  for (i=0; i<array_size[0]; ++i)
    PyTuple_SET_ITEM(array, i, PyFloat_FromDouble(sm->svm_model->alpha[i]));
#endif
  PyObject_SetAttrString(py_sm, "alpha", array);
  Py_DECREF(array);
  /* Add the index array. */
  if (sm->svm_model->index) {
    array_size[0] = sm->svm_model->totdoc+2;
#if defined(NUMERIC)||defined(NUMARRAY)
    array = PyArray_FromDimsAndData(1, array_size, PyArray_LONG,
				    (char*)(sm->svm_model->index));
#else
    array = PyTuple_New(array_size[0]);
    for (i=0; i<array_size[0]; ++i)
      PyTuple_SET_ITEM(array, i, PyFloat_FromDouble(sm->svm_model->index[i]));
#endif
    PyObject_SetAttrString(py_sm, "index", array);
    Py_DECREF(array);
  } else {
    PyObject_SetAttrString(py_sm, "index", Py_None);
  }
  /* Make the list of support vectors. */
  i=sm->svm_model->sv_num-1;
  i = i<0 ? 0 : i;
  svectors = PyTuple_New(i);
  for (i=sm->svm_model->sv_num-2; i>=0; --i) {
    PyObject *sv = docToPythonObject(sm->svm_model->supvec[i+1]);
    //PyObject_Print(sv, stdout, 0); printf("\n");
    PyTuple_SET_ITEM(svectors, i, sv);
  }
  //PyObject_Print(svectors, stdout, 0); printf("\n");
  PyObject_SetAttrString(py_sm, "supvec", svectors);
  Py_DECREF(svectors);
  
  return py_sm;
}

/************* PYTHON SVM MODULE FUNCTIONS *********/

static PyObject* emb_classify_example(PyObject *self, PyObject *args) {
  PyObject *pSm, *pSv, *cobject, *score;
  STRUCTMODEL *sm;
  DOC doc;
  PyArg_ParseTuple(args, "OO", &pSm, &pSv);
  /* Get the REAL struct model. */
  if (!PyObject_HasAttrString(pSm, "cobj")) {
    fprintf(stderr, "First arg does not appear to be a struct model, "
	    "or the cobj gone.\n");
    Py_Exit(1);
  }
  cobject = PyObject_GetAttrString(pSm, "cobj");
  if (!PyCObject_Check(cobject)) {
    fprintf(stderr, "The cobj attribute is not a CObject.\n");
    Py_Exit(1);
  }
  sm = (STRUCTMODEL*)PyCObject_AsVoidPtr(cobject);
  Py_DECREF(cobject);
  if (!sm->svm_model) {
    fprintf(stderr, "Classify_example called before svm model created!\n");
    // Don't fail outright ... just return 0.
    return PyFloat_FromDouble(0.);
  }
  /* Get the support vector. */
  if (!(doc.fvec = pythonObjectToSV(pSv))) {
    fprintf(stderr, "Second arg does not appear to be a feature vector.\n");
    Py_Exit(1);
  }
  score = PyFloat_FromDouble(classify_example(sm->svm_model, &doc));
  free_svector(doc.fvec);
  return score;
}

static PyObject* emb_create_svector(PyObject *self, PyObject *args) {
  PyObject *words, *sv, *item;
  char *userdefined="";
  double factor=1.0;
  long kernel_id=0;
  
  int i, n, isNumbers=0;
  if (!PyArg_ParseTuple(args,"O|sdl:create_svector",
			&words,&userdefined,&factor,&kernel_id)) {
    return NULL;
  }
  sv = pythonCreateBlank();
  //PyObject_SetAttrString(sv, "words", words);
  if ((n=PySequence_Size(words)) > 0) {
    item = PySequence_GetItem(words, 0);
    isNumbers = PyNumber_Check(item);
    Py_DECREF(item);
  }
  if (isNumbers) {
    /* This is a list of numbers. */
    PyObject *new_words = PyTuple_New(n-pStartingIndex);
    for (i=pStartingIndex; i<n; ++i) {
      PyObject *tuple = PyTuple_New(2);
      PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong(i));
      PyTuple_SET_ITEM(tuple, 1, PySequence_GetItem(words, i));
      PyTuple_SET_ITEM(new_words, i-pStartingIndex, tuple);
    }
    PyObject_SetAttrString(sv, "words", new_words);
    Py_DECREF(new_words);
  } else {
    /* This is a list of sequences, I guess. */
    PyObject_SetAttrString(sv, "words", words);
  }

  pythonSetS(sv, "userdefined", userdefined);
  pythonSetF(sv, "factor", factor);
  pythonSetI(sv, "kernel_id", kernel_id);
  return sv;
}

static PyObject* emb_create_doc(PyObject *self, PyObject *args) {
  PyObject *py_svec, *py_doc;
  long docnum=0xdeadbeef, slackid=0xdeadbeef, num_args;
  double costfactor=1.0;
  if (!PyArg_ParseTuple(args,"O|dll:create_doc",
			&py_svec,&costfactor,&slackid,&docnum)) {
    return NULL;
  }
  //PyObject_Print(py_svec, stdout, 0); printf("\n");
  num_args = PySequence_Size(args);
  py_doc = pythonCreateBlank();
  /* Set the support vector, and cost factor. */
  if (PySequence_Check(py_svec)) {
    PyObject_SetAttrString(py_doc, "fvec", py_svec);
  } else {
    PyObject *temp = PyTuple_New(1);
    Py_INCREF(py_svec); // The reference is stolen, so we must increment.
    PyTuple_SET_ITEM(temp, 0, py_svec);
    PyObject_SetAttrString(py_doc, "fvec", temp);
  }
  pythonSetF(py_doc, "costfactor", costfactor);
  /* Set slack ID and perhaps even document number. */
  if (num_args >= 3) {
    pythonSetI(py_doc, "slackid", slackid);
  } else {
    PyObject_SetAttrString(py_doc, "slackid", Py_None);
  }
  if (num_args == 4) {
    pythonSetI(py_doc, "docnum", docnum);
  } else {
    PyObject_SetAttrString(py_doc, "docnum", Py_None);
  }
  return py_doc;
}

static PyObject* emb_kernel(PyObject *self, PyObject *args) {
  PyObject *kp, *a, *b;
  DOC da, db;
  double result;
  /* Initialize the default kernel parameter. */
  KERNEL_PARM kernel_parm = {0, 3, 1.0, 1.0, 1.0, "empty"};
  kernel_parm.kernel_type = 0;
  kernel_parm.poly_degree = 3;
  kernel_parm.rbf_gamma = 1.0;
  kernel_parm.coef_lin = 1.0;
  kernel_parm.coef_const = 1.0;
  strcpy(kernel_parm.custom, "empty");

  if (!PyArg_ParseTuple(args, "OOO", &kp, &a, &b)) {
    return NULL;
  }
  /* Extract the support vectors. */
  da.fvec = pythonObjectToSV(a);
  if (!da.fvec) {
    fprintf(stderr, "First document does not appear to be support vector!\n");
    Py_Exit(1);
  }
  db.fvec = pythonObjectToSV(b);
  if (!db.fvec) {
    fprintf(stderr, "Second document does not appear to be support vector!\n");
    free_svector(da.fvec);
    Py_Exit(1);
  }
  /* Copy over the kernel parameters, if possible. */
  pythonGetI(kp, "kernel_type", &kernel_parm.kernel_type);
  pythonGetI(kp, "poly_degree", &kernel_parm.poly_degree);
  pythonGetF(kp, "rbf_gamma", &kernel_parm.rbf_gamma);
  pythonGetF(kp, "coef_lin", &kernel_parm.coef_lin);
  pythonGetF(kp, "coef_const", &kernel_parm.coef_const);
  pythonGetS(kp, "custom", kernel_parm.custom, sizeof(kernel_parm.custom));
  /* Call that function! */
  result = kernel(&kernel_parm, &da, &db);
  /* Get rid of the allocated support vectors. */
  free_svector(da.fvec);
  free_svector(db.fvec);
  /* Return the result. */
  return PyFloat_FromDouble(result);
}

/************ SVM STRUCT FUNCTIONS **********/

SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples;
  long i;
  PyObject *pDict, *pFunc, *pArgs, *pValue;

  /* Call the Python function read_examples in the modules. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_READ_EXAMPLES);
  if (pFunc == NULL) {
    fprintf(stderr, "Could not find function %s!\n", PYTHON_READ_EXAMPLES);
    Py_Exit(1);
  }
  pArgs = PyTuple_New(2); /* This is a new instance! */
  pValue = PyString_FromString(file);
  PyTuple_SetItem(pArgs, 0, pValue);
  PyTuple_SetItem(pArgs, 1, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PYTHON_CALL(pValue, pFunc, pArgs);
  /* Make sure that it's a sequence. */
  Py_DECREF(pArgs);
  if (pValue == NULL) {
    fprintf(stderr, "%s function failed!\n", PYTHON_READ_EXAMPLES);
    Py_Exit(1);
  }
  if (!PySequence_Check(pValue)) {
    fprintf(stderr, "%s function did not return a sequence!\n",
	    PYTHON_READ_EXAMPLES);
    Py_Exit(1);
  }
  /* Everything's checked out sofar.  Retrieve the result. */
  //PyObject_Print(pValue, stdout, 0); printf("\n");
  sample.n = PySequence_Size(pValue);
  examples=(EXAMPLE *)my_malloc(sizeof(EXAMPLE)*sample.n);
  for (i=0; i<sample.n; ++i) {
    PyObject *pExample = PySequence_GetItem(pValue, i);
    if (!pExample || !PySequence_Check(pExample) ||
	PySequence_Size(pExample)<2){
      fprintf(stderr, "%s's item %ld is not a sequence element of "
	      "at least two items!\n", PYTHON_READ_EXAMPLES, i);
      Py_Exit(1);
    }
    examples[i].x.py_pattern = PySequence_GetItem(pExample, 0);
    examples[i].y.py_label   = PySequence_GetItem(pExample, 1);
    Py_DECREF(pExample);
  }
  Py_DECREF(pValue);

  /* Store the result, and get the hell out of Dodge. */
  sample.examples=examples;

  return(sample);
}

void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm)
{
  /* Initialize structmodel sm. The weight vector w does not need to be
     initialized, but you need to provide the maximum size of the
     feature space in sizePsi. This is the maximum number of different
     weights that can be learned. Later, the weight vector w will
     contain the learned weights for the model. */
  PyObject *pDict, *pFunc, *pArgs, *pValue, *py_sm;

  sm->dirty = 1;
  sm->svm_model=sm->py_sm=NULL;
  sm->sizePsi=0;
  /* Set up the call the Python function parse_parameters in the module. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_INIT_MODEL);
  if (pFunc == NULL) {
    fprintf(stderr, "Could not find function %s!\n",
	    PYTHON_INIT_MODEL);
    Py_Exit(1);
  }
  /* Build the argument list. */
  pArgs = PyTuple_New(3);
  PyTuple_SetItem(pArgs, 0, sampleToPythonObject(sample));
  PyTuple_SetItem(pArgs, 1, smToPythonObject(sm));
  PyTuple_SetItem(pArgs, 2, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PYTHON_CALL(pValue, pFunc, pArgs);
  Py_DECREF(pArgs);
  Py_DECREF(pValue);
  /* Sets sizePsi to the value set to sm.size_psi in the python
     function. */
  py_sm = smToPythonObject(sm);
  pValue = PyObject_GetAttrString(py_sm, "size_psi");
  sm->sizePsi = PyInt_AsLong(pValue);
  //printf("sizePsi as %ld\n", sm->sizePsi);
  Py_DECREF(py_sm);
  Py_DECREF(pValue);
  /* Some structures may need to be reinitialized after this. */
  sm->dirty = 1;
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Initializes the optimization problem. Typically, you do not need
     to change this function, since you want to start with an empty
     set of constraints. However, if for example you have constraints
     that certain weights need to be positive, you might put that in
     here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
     is an array of feature vectors, rhs is an array of doubles. m is
     the number of constraints. The function returns the initial
     set of constraints. */
  CONSTSET c;
  long     sizePsi=sm->sizePsi;
  long     i;
  WORD     words[2];
  PyObject *pDict, *pFunc, *pArgs, *pValue, *pItem, *pFirst, *pSecond;

  if(1) { /* normal case: start with empty set of constraints */
    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;
  }
  else { /* add constraints so that all learned weights are
            positive. WARNING: Currently, they are positive only up to
            precision epsilon set by -e. */
    c.lhs=my_malloc(sizeof(DOC *)*sizePsi);
    c.rhs=my_malloc(sizeof(double)*sizePsi);
    for(i=0; i<sizePsi; i++) {
      words[0].wnum=i+1;
      words[0].weight=1.0;
      words[1].wnum=0;
      /* the following slackid is a hack. we will run into problems,
         if we have move than 1000000 slack sets (ie examples) */
      c.lhs[i]=create_example(i,0,1000000+i,1,create_svector(words,"",1.0));
      c.rhs[i]=0.0;
    }
  }

  /* Set up the call the Python function. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_INIT_CONSTRAINTS);
  if (pFunc == NULL) {
    /*fprintf(stderr, "Could not find function %s!\n",
      PYTHON_INIT_CONSTRAINTS);*/
    return c;
  }
  /* Build the argument list. */
  pArgs = PyTuple_New(3);
  PyTuple_SetItem(pArgs, 0, sampleToPythonObject(sample));
  PyTuple_SetItem(pArgs, 1, smToPythonObject(sm));
  PyTuple_SetItem(pArgs, 2, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PYTHON_CALL(pValue, pFunc, pArgs);
  Py_DECREF(pArgs);
  /* Process the list of constraints. */
  if (!pValue) {
    fprintf(stderr, "Badness happend in %s.\n",
	    PYTHON_INIT_CONSTRAINTS);
    Py_Exit(1);
  }
  if (pValue == Py_None) {
    return c;
  }
  if (!PySequence_Check(pValue)) {
    fprintf(stderr, "Object from %s is not None or a sequence.\n",
	    PYTHON_INIT_CONSTRAINTS);
    Py_Exit(1);
  }
  /* Make the constraints. */
  c.m = PySequence_Size(pValue);
  c.lhs=(DOC**)my_malloc(sizeof(DOC *)*c.m);
  c.rhs=(double*)my_malloc(sizeof(double)*c.m);
  for (i=0; i<c.m; ++i) {
    /* Make one constraint! */
    DOC *doc;
    pItem = PySequence_GetItem(pValue, i);
    pFirst=PySequence_GetItem(pItem, 0);
    pSecond=PySequence_GetItem(pItem, 1);
    if (!pSecond || !pFirst) {
      fprintf(stderr, "Item %ld in sequence from %s doesn't have 2 items.\n",
	      i, PYTHON_INIT_CONSTRAINTS);
      Py_Exit(1);
    }
    /* Make the right hand side of the constriant. */
    if (!PyNumber_Check(pSecond)) {
      fprintf(stderr, "Second item if item %ld in sequence from %s not a "
	      "number.\n", i, PYTHON_INIT_CONSTRAINTS);
      Py_Exit(1);
    }
    c.rhs[i] = PyFloat_AsDouble(pSecond);
    /* Make the left hand side of the constraint. */
    doc = pythonObjectToDoc(pFirst);
    if (!doc) {
      fprintf(stderr, "First item of item %ld in sequence from %s not "
	      "a doc.\n", i, PYTHON_INIT_CONSTRAINTS);
      Py_Exit(1);
    }
    /* Some things must be set. */
    doc->docnum = doc->kernelid = i;
    if (doc->slackid == 0xdeadbeef) doc->slackid = sample.n+i;
    c.lhs[i] = doc;
  }
  Py_DECREF(pValue);

  return(c);
}

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label yhat for pattern x that scores the highest
     according to the linear evaluation function in sm, especially the
     weights sm.w. The returned label is taken as the prediction of sm
     for the pattern x. The weights correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. If the
     function cannot find a label, it shall return an empty label as
     recognized by the function empty_label(y). */
  LABEL y;
  /* insert your code for computing the predicted label y here */
  PyObject *pDict, *pFunc, *pArgs;
  /* Set up the call the Python function. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_CLASSIFY_EXAMPLE);
  if (pFunc == NULL) {
    fprintf(stderr, "Could not find function %s!\n", PYTHON_CLASSIFY_EXAMPLE);
    Py_Exit(1);
  }
  pArgs = PyTuple_New(3); /* This is a new instance! */
  Py_INCREF((PyObject*)x.py_pattern);
  PyTuple_SetItem(pArgs, 0, (PyObject*)x.py_pattern);
  PyTuple_SetItem(pArgs, 1, smToPythonObject(sm));
  PyTuple_SetItem(pArgs, 2, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PYTHON_CALL(y.py_label, pFunc, pArgs);
  Py_DECREF(pArgs);

  return(y);
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the slack rescaling
     formulation. It has to take into account the scoring function in
     sm, especially the weights sm.w, as well as the loss
     function. The weights in sm.w correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. Most simple
     is the case of the zero/one loss function. For the zero/one loss,
     this function should return the highest scoring label ybar, if
     ybar is unequal y; if it is equal to the correct label y, then
     the function shall return the second highest scoring label. If
     the function cannot find a label, it shall return an empty label
     as recognized by the function empty_label(y). */
  LABEL ybar;
  /* insert your code for computing the label ybar here */
  PyObject *pDict, *pFunc, *pArgs;
  static int has_not_warned = 1;
  /* Set up the call the Python function. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_FIND_MOST_VIOLATED);
  if (pFunc == NULL) {
    if (has_not_warned) {
      fprintf(stderr, "Warning: could not find function %s!\n",
	      PYTHON_FIND_MOST_VIOLATED);
      has_not_warned = 0;
    }
    return classify_struct_example(x, sm, sparm);
  }

  pArgs = PyTuple_New(4); /* This is a new instance! */
  Py_INCREF((PyObject*)x.py_pattern);
  Py_INCREF((PyObject*)y.py_label);
  PyTuple_SetItem(pArgs, 0, (PyObject*)x.py_pattern);
  PyTuple_SetItem(pArgs, 1, (PyObject*)y.py_label);
  PyTuple_SetItem(pArgs, 2, smToPythonObject(sm));
  PyTuple_SetItem(pArgs, 3, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PYTHON_CALL(ybar.py_label, pFunc, pArgs);
  Py_DECREF(pArgs);
  
  return(ybar);
}

LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the margin rescaling
     formulation. It has to take into account the scoring function in
     sm, especially the weights sm.w, as well as the loss
     function. The weights in sm.w correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. Most simple
     is the case of the zero/one loss function. For the zero/one loss,
     this function should return the highest scoring label ybar, if
     ybar is unequal y; if it is equal to the correct label y, then
     the function shall return the second highest scoring label. If
     the function cannot find a label, it shall return an empty label
     as recognized by the function empty_label(y). */
  /* insert your code for computing the label ybar here */
  return find_most_violated_constraint_slackrescaling(x, y, sm, sparm);
}

int         empty_label(LABEL y)
{
  /* Returns true, if y is an empty label. An empty label might be
     returned by find_most_violated_constraint_???(x, y, sm) if there
     is no incorrect label that can be found for x, or if it is unable
     to label x at all */
  return y.py_label == Py_None;
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
		 STRUCT_LEARN_PARM *sparm)
{
  /* Returns a feature vector describing the match between pattern x
     and label y. The feature vector is returned as a list of
     SVECTOR's. Each SVECTOR is in a sparse representation of pairs
     <featurenumber:featurevalue>, where the last pair has
     featurenumber 0 as a terminator. Featurenumbers start with 1 and
     end with sizePsi. Featuresnumbers that are not specified default
     to value 0. As mentioned before, psi() actually returns a list of
     SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
     specifies the next element in the list, terminated by a NULL
     pointer. The list can be though of as a linear combination of
     vectors, where each vector is weighted by its 'factor'. This
     linear combination of feature vectors is multiplied with the
     learned (kernelized) weight vector to score label y for pattern
     x. Without kernels, there will be one weight in sm.w for each
     feature. Note that psi has to match
     find_most_violated_constraint_???(x, y, sm) and vice versa. In
     particular, find_most_violated_constraint_???(x, y, sm) finds
     that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
     inner vector product) and the appropriate function of the
     loss + margin/slack rescaling method. See that paper for details. */
  SVECTOR *fvec;

  /* insert code for computing the feature vector for x and y here */
  PyObject *pDict, *pFunc, *pArgs, *pPsi;
  /* Set up the call the Python function. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_PSI);
  if (pFunc == NULL) {
    fprintf(stderr, "Could not find function %s!\n", PYTHON_PSI);
    Py_Exit(1);
  }

  pArgs = PyTuple_New(4); /* This is a new instance! */
  Py_INCREF((PyObject*)x.py_pattern);
  Py_INCREF((PyObject*)y.py_label);
  PyTuple_SetItem(pArgs, 0, (PyObject*)x.py_pattern);
  PyTuple_SetItem(pArgs, 1, (PyObject*)y.py_label);
  PyTuple_SetItem(pArgs, 2, smToPythonObject(sm));
  PyTuple_SetItem(pArgs, 3, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PYTHON_CALL(pPsi, pFunc, pArgs);
  Py_DECREF(pArgs);
  /* Convert this to a support vector. */
  if (!(fvec = pythonObjectToSV(pPsi))) {
    fprintf(stderr, "Function %s returned a bad value!\n", PYTHON_PSI);
    Py_Exit(1);
  }
  Py_DECREF(pPsi);
  return(fvec);
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  static int has_not_warned = 1;
  PyObject *pDict, *pFunc, *pArgs, *pLoss;
  double theLoss;
  /* Set up the call the Python function. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_LOSS);
  if (pFunc == NULL) {
    /* We could not find the loss function! */
    int lossResult;
    if (has_not_warned) {
      fprintf(stderr, "Warning: %s function missing.  Defaulting to 0/1 based "
	      "on equality.", PYTHON_LOSS);
      has_not_warned = 0;
    }
    /* Perform 0/1 loss based on Python object comparison.  Equality
       is pretty good for everything except custom classes that don't
       specify how to handle the equality operator. */
    lossResult=PyObject_RichCompareBool(y.py_label, ybar.py_label, Py_EQ);
    return (lossResult == 0) ? 1 : 0;
  }
  /* We have a loss function available?  Great.  Let's use it! */
  pArgs = PyTuple_New(3); /* This is a new instance! */
  Py_INCREF((PyObject*)y.py_label);
  Py_INCREF((PyObject*)ybar.py_label);
  PyTuple_SetItem(pArgs, 0, (PyObject*)y.py_label);
  PyTuple_SetItem(pArgs, 1, (PyObject*)ybar.py_label);
  PyTuple_SetItem(pArgs, 2, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PyErr_Clear();
  PYTHON_CALL(pLoss, pFunc, pArgs);
  if (PyErr_Occurred()) {
    PyErr_Print();
    Py_Exit(1);
  }
  Py_DECREF(pArgs);
  theLoss = PyFloat_AsDouble(pLoss);
  Py_DECREF(pLoss);
  return theLoss;
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm)
{
  /* This function is called after training and allows final touches to
     the model sm. But primarly it allows computing and printing any
     kind of statistic (e.g. training error) you might want. */
  int i;
  PyObject *pDict, *pFunc, *pArgs, *pValue, *pConsts, *pAlpha;
  /* Set up the call to the Python method. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_PRINT_LEARNING_STATS);
  if (pFunc == NULL) {
    /*fprintf(stderr, "Warning: could not find function %s!\n",
      PYTHON_PRINT_LEARNING_STATS);*/
    /* Do default behavior of outputting a list of the loss for all
       training examples under the current model. */
    printf("[");
    for (i=0; i<sample.n; i++) {
      LABEL ypred = classify_struct_example(sample.examples[i].x, sm, sparm);
      double trainingLoss = loss(sample.examples[i].y, ypred, sparm);
      if (i) printf(", "); else printf(" ");
      printf("%g", trainingLoss);
      free_label(ypred);
    }
    printf(" ]\n");
    return;
  }
  /* Build the constraint list. */
  pConsts = PyTuple_New(cset.m);
  for (i=0; i<cset.m; ++i) {
    PyObject *constraint = PyTuple_New(2);
    PyTuple_SET_ITEM(constraint, 0, svToPythonObject(cset.lhs[i]->fvec));
    PyTuple_SET_ITEM(constraint, 1, PyFloat_FromDouble(cset.rhs[i]));
    PyTuple_SET_ITEM(pConsts, i, constraint);
  }
  /* Build alpha. */
#if defined(NUMERIC)||defined(NUMARRAY)
  pAlpha = PyArray_FromDimsAndData(1, (int*)&cset.m, PyArray_DOUBLE,
				   (char*)alpha);
#else
  pAlpha = PyTuple_New(cset.m);
  for (i=0; i<cset.m; ++i)
    PyTuple_SET_ITEM(pAlpha, i, PyFloat_FromDouble(alpha[i]));
#endif
  /* Build the argument list. */
  pArgs = PyTuple_New(5);
  PyTuple_SET_ITEM(pArgs, 0, sampleToPythonObject(sample));
  PyTuple_SET_ITEM(pArgs, 1, smToPythonObject(sm));
  PyTuple_SET_ITEM(pArgs, 2, pConsts);
  PyTuple_SET_ITEM(pArgs, 3, pAlpha);
  PyTuple_SET_ITEM(pArgs, 4, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PyErr_Clear();
  PYTHON_CALL(pValue, pFunc, pArgs);
  if (PyErr_Occurred()) {
    PyErr_Print();
    Py_Exit(1);
  }
  Py_DECREF(pArgs);
  Py_DECREF(pValue);
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm, 
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */
  PyObject *pDict, *pFunc, *pArgs, *pValue;
  if (sample.n==0) {
    /* If we have no test examples then py_stats would never have been
       initialized, even to None. */
    teststats->py_stats = Py_None;
  }
  /* Set up the call to the Python method. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_PRINT_TESTING_STATS);
  if (pFunc == NULL) {
    /*fprintf(stderr, "Warning: could not find function %s!\n",
      PYTHON_PRINT_TESTING_STATS);*/
    /* Do default behavior of printing out the teststats->py_stats object. */
    PyObject_Print((PyObject*)teststats->py_stats, stdout, 0);
    printf("\n");
    return;
  }
  /* Build the argument list. */
  pArgs = PyTuple_New(4);
  Py_INCREF((PyObject*)teststats->py_stats);
  PyTuple_SetItem(pArgs, 0, sampleToPythonObject(sample));
  PyTuple_SetItem(pArgs, 1, smToPythonObject(sm));
  PyTuple_SetItem(pArgs, 2, sparmToPythonObject(sparm));
  PyTuple_SetItem(pArgs, 3, (PyObject*)teststats->py_stats);
  /* Call the embedded python function!! */
  PYTHON_CALL(pValue, pFunc, pArgs);
  Py_DECREF(pArgs);
  Py_DECREF(pValue);
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
			    STRUCT_TEST_STATS *teststats)
{
  /* This function allows you to accumlate statistic for how well the
     predicition matches the labeled example. It is called from
     svm_struct_classify. See also the function
     print_struct_testing_stats. */
  PyObject *pDict, *pFunc, *pArgs;
  if(exnum == 0) { /* this is the first time the function is
		      called. So initialize the teststats */
    teststats->py_stats = Py_None;
  }
  /* Set up the call to the Python method. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_EVAL_PREDICTION);
  if (pFunc == NULL) {
    PyObject *pList, *pNumber;
    /*fprintf(stderr, "Warning: could not find function %s!\n",
      PYTHON_EVAL_PREDICTION);*/
    /* Do default behavior of accumulating the loss in a list. */
    if (exnum == 0) {
      teststats->py_stats = PyList_New(0);
    }
    pList = (PyObject*)teststats->py_stats;
    pNumber = PyFloat_FromDouble(loss(ex.y, ypred, sparm));
    PyList_Append(pList, pNumber);
    Py_DECREF(pNumber);
    return;
  }
  /* Build the argument list. */
  pArgs = PyTuple_New(7);
  PyTuple_SetItem(pArgs, 0, PyInt_FromLong(exnum));
  Py_INCREF((PyObject*)ex.x.py_pattern);
  Py_INCREF((PyObject*)ex.y.py_label);
  Py_INCREF((PyObject*)ypred.py_label);
  Py_INCREF((PyObject*)teststats->py_stats);
  PyTuple_SetItem(pArgs, 1, (PyObject*)ex.x.py_pattern);
  PyTuple_SetItem(pArgs, 2, (PyObject*)ex.y.py_label);
  PyTuple_SetItem(pArgs, 3, (PyObject*)ypred.py_label);
  PyTuple_SetItem(pArgs, 4, smToPythonObject(sm));
  PyTuple_SetItem(pArgs, 5, sparmToPythonObject(sparm));
  PyTuple_SetItem(pArgs, 6, (PyObject*)teststats->py_stats);
  /* Call the embedded python function!! */
  PyErr_Clear();
  PYTHON_CALL(teststats->py_stats, pFunc, pArgs);
  if (PyErr_Occurred()) {
    PyErr_Print();
    Py_Exit(1);
  }

  Py_DECREF(pArgs);
}

void        write_struct_model(char *file, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* Writes structural model sm to file file. */
  PyObject *pDict, *pFunc, *pArgs, *pValue, *pFile, *module, *pSm, *pArray;

  pSm = smToPythonObject(sm);
  pArray = PyObject_GetAttrString(pSm, "w");
  pValue = PySequence_Tuple(pArray);
  Py_DECREF(pArray);
  PyObject_SetAttrString(pSm, "w", pValue);
  Py_DECREF(pValue);

  pArray = PyObject_GetAttrString(pSm, "alpha");
  if (pArray) {
    pValue = PySequence_Tuple(pArray);
    Py_DECREF(pArray);
    PyObject_SetAttrString(pSm, "alpha", pValue);
    Py_DECREF(pValue);
  }

  pArray = PyObject_GetAttrString(pSm, "index");
  if (pArray && pArray != Py_None) {
    pValue = PySequence_Tuple(pArray);
    Py_DECREF(pArray);
    PyObject_SetAttrString(pSm, "index", pValue);
    Py_DECREF(pValue);
  }
  
  if (PyObject_HasAttrString(pSm, "cobj"))
    PyObject_DelAttrString(pSm, "cobj");

  /* Set up the call to the Python method. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_WRITE_MODEL);
  if (pFunc == NULL) {
    /*fprintf(stderr, "Warning: could not find function %s!\n",
      PYTHON_WRITE_MODEL);*/
    /* Do the default behavior of using pickle to dump the structmodel
       to a file. */
    module = PyImport_ImportModule("pickle");
    if (module == NULL) {
      fprintf(stderr, "The cPickle package is unavailable!!\n");
      Py_Exit(1);
    }
    pFunc = PyDict_GetItemString(PyModule_GetDict(module), "dump");
    /* Open the file for writing. */
    pFile = PyFile_FromString(file, "w");
    pArgs = Py_BuildValue("(OOi)", pSm, pFile, -1);
    /* Dump it! */
    PYTHON_CALL(pValue, pFunc, pArgs);
    if (pValue == NULL) {
      fprintf(stderr, "cPickle.dump failed to write structmodel!\n");
      Py_Exit(1);
    }
    Py_DECREF(pValue);
    Py_DECREF(pArgs);
    Py_DECREF(pFile);
    Py_DECREF(module);
    return;
  }
  /* Build the argument list. */
  pArgs = PyTuple_New(3);
  PyTuple_SetItem(pArgs, 0, PyString_FromString(file));
  PyTuple_SetItem(pArgs, 1, pSm);
  PyTuple_SetItem(pArgs, 2, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PYTHON_CALL(pValue, pFunc, pArgs);
  Py_DECREF(pArgs);
  Py_DECREF(pValue);
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
  /* Writes structural model sm to file file. */
  PyObject *pDict, *pFunc, *pArgs, *pValue, *pFile, *module, *pSm;
  STRUCTMODEL sm;
  int i, n;

  /* Do some minimal initialization of sparm. */
  sparm->py_sparm = NULL;
  sparm->custom_argc = 0;

  /* Set up the call to the Python method. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_READ_MODEL);
  if (pFunc == NULL) {
    /*fprintf(stderr, "Warning: could not find function %s!\n",
      PYTHON_READ_MODEL);*/
    /* Do the default behavior of using pickle to load the structmodel
       from a file. */
    module = PyImport_ImportModule("pickle");
    if (module == NULL) {
      fprintf(stderr, "The pickle package is unavailable!!\n");
      Py_Exit(1);
    }
    pFunc = PyDict_GetItemString(PyModule_GetDict(module), "load");
    /* Open the file for reading. */
    pFile = PyFile_FromString(file, "r");
    pArgs = Py_BuildValue("(O)", pFile);
    /* Load it! */
    PYTHON_CALL(pSm, pFunc, pArgs);
    if (pSm == NULL) {
      fprintf(stderr, "pickle.load failed to read structmodel!\n");
      Py_Exit(1);
    }
    Py_DECREF(pFile);
    Py_DECREF(module);
  } else {
    /* Build the argument list. */
    pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyString_FromString(file));
    PyTuple_SetItem(pArgs, 1, sparmToPythonObject(sparm));
    /* Call the embedded python function!! */
    PYTHON_CALL(pSm, pFunc, pArgs);
    if (pSm == Py_None) {
      fprintf(stderr, "%s returned None, indicating a read error!\n",
	      PYTHON_READ_MODEL);
      Py_Exit(1);
    }
  }
  Py_DECREF(pArgs);
  /* By whatever method, either the user function or default behavior,
     we now apparently have a Python struct model object.  Put the
     relevant fields from the returned value in the C structmodel. */
  sm.py_sm = pSm;
  sm.dirty = 1;
  sm.svm_model = (MODEL*)my_malloc(sizeof(MODEL));
  /* Set scalar parameters. */
  assert(pythonGetI(pSm, "size_psi", &sm.sizePsi));
  /* Set scalar parameters for the model. */
  assert(pythonGetI(pSm, "sv_num", &sm.svm_model->sv_num));
  assert(pythonGetF(pSm, "b", &sm.svm_model->b));
  assert(pythonGetI(pSm, "totwords", &sm.svm_model->totwords));
  assert(pythonGetI(pSm, "totdoc", &sm.svm_model->totdoc));
  assert(pythonGetF(pSm, "loo_error", &sm.svm_model->loo_error));
  assert(pythonGetF(pSm, "loo_recall", &sm.svm_model->loo_recall));
  assert(pythonGetF(pSm, "loo_precision", &sm.svm_model->loo_precision));
  assert(pythonGetF(pSm, "xa_error", &sm.svm_model->xa_error));
  assert(pythonGetF(pSm, "xa_recall", &sm.svm_model->xa_recall));
  assert(pythonGetF(pSm, "xa_precision", &sm.svm_model->xa_precision));
  assert(pythonGetF(pSm, "maxdiff", &sm.svm_model->maxdiff));
  assert(pythonGetI(pSm, "at_upper_bound", &sm.svm_model->at_upper_bound));
  /* Set scalar parameters from the kernel parameter. */
  assert(pythonGetI(pSm,"kernel_type",&sm.svm_model->kernel_parm.kernel_type));
  assert(pythonGetI(pSm,"poly_degree",&sm.svm_model->kernel_parm.poly_degree));
  assert(pythonGetF(pSm, "rbf_gamma", &sm.svm_model->kernel_parm.rbf_gamma));
  assert(pythonGetF(pSm, "coef_lin", &sm.svm_model->kernel_parm.coef_lin));
  assert(pythonGetF(pSm, "coef_const", &sm.svm_model->kernel_parm.coef_const));
  assert(pythonGetS(pSm, "custom", sm.svm_model->kernel_parm.custom,
		    sizeof(sm.svm_model->kernel_parm.custom)));
  /* Copy over the friggin' w array! */
  pValue = PyObject_GetAttrString(pSm, "w");
  if (pValue == NULL) {
    fprintf(stderr, "No w attribute for sm read from %s!", file);
    Py_Exit(1);
  }
  sm.svm_model->lin_weights = sm.w =
    (double*)my_malloc((sm.sizePsi+1)*sizeof(double));
  n = PySequence_Size(pValue);
  if (n > sm.sizePsi) n=sm.sizePsi;
  for (i=0+pStartingIndex; i<n+pStartingIndex; ++i) {
    PyObject *pPsiValue = PySequence_GetItem(pValue, i);
    sm.w[i+1-pStartingIndex] = PyFloat_AsDouble(pPsiValue);
    Py_DECREF(pPsiValue);
  }
  Py_DECREF(pValue);
  /* Copy over the support vectors. */
  assert(pValue = PyObject_GetAttrString(pSm, "supvec"));
  sm.svm_model->supvec = (DOC**)my_malloc(sm.svm_model->sv_num*sizeof(DOC*));
  n = PySequence_Size(pValue)+1;
  if (n > sm.svm_model->sv_num) n=sm.svm_model->sv_num;
  if (n > 0)
    sm.svm_model->supvec[0] = NULL;
  for (i=1; i<n; ++i) {
    PyObject *pSv = PySequence_GetItem(pValue, i-1);
    DOC *sv = pythonObjectToDoc(pSv);
    Py_DECREF(pSv);
    sm.svm_model->supvec[i] = sv;
  }
  Py_DECREF(pValue);
  /* Copy over the alpha array. */
  assert(pValue = PyObject_GetAttrString(pSm, "alpha"));
  n = PySequence_Size(pValue);
  sm.svm_model->alpha = (double*)my_malloc(n*sizeof(double));
  for (i=0; i<n; ++i) {
    PyObject *pItem = PySequence_GetItem(pValue, i);
    sm.svm_model->alpha[i] = PyFloat_AsDouble(pItem);
    Py_DECREF(pItem);
  }
  Py_DECREF(pValue);
  /* Copy over the index array. */
  assert(pValue = PyObject_GetAttrString(pSm, "index"));
  if (pValue == Py_None) {
    sm.svm_model->index = NULL;
  } else {
    n = PySequence_Size(pValue);
    sm.svm_model->index = (long*)my_malloc(n*sizeof(long));
    for (i=0; i<n; ++i) {
      PyObject *pItem = PySequence_GetItem(pValue, i);
      sm.svm_model->index[i] = PyInt_AsLong(pItem);
      Py_DECREF(pItem);
    }
    Py_DECREF(pValue);
  }
  /* That'll do, pig.  That'll do. */
  return sm;
}

int pythonDummyClose(FILE*f) {
  /* This function does nothing, since SVM^struct handles the opening
     and closing of the file by itself very well, thank you.  It is
     only used because we need to pass an already open FILE into
     Python, and the function requires a function. */
  return 0;
}

void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
  PyObject *pDict, *pFunc, *pArgs, *pValue;
  /* Set up the call to the Python method. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_WRITE_LABEL);
  if (pFunc == NULL) {
    /*fprintf(stderr, "Warning: could not find function %s!\n",
      PYTHON_WRITE_LABEL);*/
    /* Perform default behavior of simply outputting the label to the file. */
    PyObject_Print((PyObject*)y.py_label, fp, 0);
    fprintf(fp, "\n");
    return;
  }
  /* Build the argument list. */
  Py_INCREF((PyObject*)y.py_label);
  pArgs = PyTuple_New(2);
  PyTuple_SetItem(pArgs, 0, PyFile_FromFile(fp,"foobar","w",pythonDummyClose));
  PyTuple_SetItem(pArgs, 1, (PyObject*)y.py_label);
  /* Call the embedded python function!! */
  PYTHON_CALL(pValue, pFunc, pArgs);
  Py_DECREF(pArgs);
  Py_DECREF(pValue);
}

void        free_pattern(PATTERN x) {
  /* Frees the memory of x. */
  Py_DECREF((PyObject*)x.py_pattern);
}

void        free_label(LABEL y) {
  /* Frees the memory of y. */
  Py_DECREF((PyObject*)y.py_label);
}

void        free_struct_model(STRUCTMODEL sm) 
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
  Py_DECREF((PyObject*)sm.py_sm);
}

void        free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++) { 
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void        print_struct_help()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_learn. */
  /* Writes label y to file handle fp. */
  PyObject *pDict, *pFunc, *pArgs, *pValue;
  /* Set up the call to the Python method. */
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_PRINT_HELP);
  if (pFunc == NULL) {
    /*fprintf(stderr, "Warning: could not find function %s!\n",
      PYTHON_PRINT_HELP);*/
    /* Perform default behavior of simply outputting SVM^struct's
       regular help string. */
    printf
      ("         --* string  -> custom parameters that can be adapted for\n"
       "                        struct learning. The * can be replaced by\n"
       "                        any string and there can be multiple\n"
       "                        options starting with --.\n"
       "\n"
       "         --m module  -> for the Python code, use the module named\n"
       "                        'module'.  In most settings, the module\n"
       "                        is stored in the Python file module.py.\n"
       "                        If omitted, the default is to load the\n"
       "                        module %s.", PYTHON_MODULE_NAME
       );
    return;
  }
  /* Build the argument list. */
  pArgs = PyTuple_New(0);
  /* Call the embedded python function!! */
  PYTHON_CALL(pValue, pFunc, pArgs);
  Py_DECREF(pArgs);
  Py_DECREF(pValue);
  
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
  PyObject *pDict, *pFunc, *pArgs, *pValue;
  char *moduleName = PYTHON_MODULE_NAME;
  int i;

  /* Detect whether we should use an alternate module name. */
  for(i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0] == '-');i++){
    if (!strcmp("--m", sparm->custom_argv[i])) {
      moduleName = sparm->custom_argv[++i];
      break;
    }
  }

  /* Load the module. */
  api_load_module(moduleName);
  /* Set up the call the Python function parse_parameters in the module. */
  sparm->py_sparm = NULL;
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict, PYTHON_PARSE_ARGUMENTS);
  if (pFunc == NULL) {
    fprintf(stderr, "Warning: could not find function %s!\n",
	    PYTHON_PARSE_ARGUMENTS);
    return;
  }
  pArgs = PyTuple_New(1); /* This is a new instance! */
  PyTuple_SetItem(pArgs, 0, sparmToPythonObject(sparm));
  /* Call the embedded python function!! */
  PYTHON_CALL(pValue, pFunc, pArgs);
  Py_DECREF(pArgs);
  Py_DECREF(pValue);
}
