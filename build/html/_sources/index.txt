.. Hmax documentation master file, created by
   sphinx-quickstart on Wed Dec 19 01:00:26 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Hmax's documentation!
================================


OS and Hardware Prerequisites:
-----------------------------
This package was developped under an ubuntu box version 12.04. I haven't tested it on a mac or a windows machine. Since it's writen in python and using cross paltform packages, it should be cross paltform as well, but again, it is not garantee.

This was tested on linux box with an NVIDIA GTX 570 and 690 GPUs. From my experience, the 570 is "faster" than the 690. Of course faster is to be taken with grain of salt since this wasn't failry benchmarked using all the features available on both GPUs. Which means that the cuda implementation is pretty simple and do not use the extra features that the 690 offers.

Installation:
-------------
To get the hmax package, you need to send an email to youssef.barhomi@gmail.com in order to get access to the private git hub repository, after confirmation, you can fetch the src code with this::

        $git clone https://github.com/serre-lab/hmax.git

Of course, this needs git to be installed on your machine (assming)::
    
        $sudo apt-get install git-core

The hmax package is writen following the python packages standard, you will need to install setup tools first from here http://pypi.python.org/pypi/setuptools, then go to the hmax source directory and type this::
    
        $sudo python setup.py install

and that should take care of all your dependencies. Now let's jump to our first demo.
Dependencies are:
* pycuda
* pytables
* shogun
* cheetah  

Run the model on Caltech 101:
-----------------------------
* Download the caltech 101 database from `here <http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz>`_
* Go to the file hmax/models/params.py where all model parameters are defined. the first model is the basic model (ventral_1), you will need to change the directory name of caltech 101 from 101_Objectcategories... to caltech_101. The code assumes your hmax is under /home/youssef/src/hmax and you caltech_101 is under /home/youssef/caltech_101. If you need to change anyhting, that would be home_dir variable and hmax_path and dir_images
* go back to your hmax root dir and do::

  $python ./run_hmax.py exp_1 ventral_1

which basically runs hmax on caltech_101 by extracting features from 15 traing images from each class (102 classes) and 15 testing images from each class (not overlapping). The generates a 1000 c1 words dictionary then extract c2 features. At the end it classifies 
* when it's done, it will print classification results 

Directory structure:
----------------------
.. image:: figures/dir_structure_2.jpeg


Design:
=======

This document is intended to describe how this package will work starting from the user perspective point of you. 

Use cases:
----------

**Use case #1:**
    running a model of gray or color images category classification
    ::
        p       = params.ventral_1()
        data    = get_input.from_images(p)    
        l_words = ventral.get_dict(data, p)
        l_c2    = ventral.get_c2(data,l_words, p)
        results = classification.svm(data,p)

**Use case #2:** 
    running the use case #1 while exploring the parameter space 

**Use case 3:** 
    running temporal model on few videos (related to the mouse project)
    PS: if parameter space exploration needed, follow #2
**Use case #4:** 
    temporal model in real time (related to locomotion project)

**Use case #5:** 
    shape hmax on ROS (PR-2 project)

**Use case #6:** 
    stereo vision processing

**Use case #7:** 
    texture vs contour experiments

**Use case #8:** 
    Contours generation from images (david's project)

**Use case #9:** 
    be able to switch to different kernels for extracting features 
    These possibilities include:
    * libjacket from accelereyes
    * theano
    * pyopencl
    * pure python
    * python gpu library from mason
    * other cuda kernels using NPP cuda libraries 

**Challenges:**
    * replace kernels easily
    * get all layers from gpu for debuging purposes
    * comparing different kernel outputs to other implementations (pycuda compared to matlab code)
    * be able to run kernels differenetly thanks to code template generation with cheetah (meta programming)
    * do "real time" processing and be able to show it (a bit like opencv)
    * be able to include easily opencv algorithms with my code

Functions definitions
-------------------------
Here is a list of few function definitions that will be the building blocks of the package. I will try to keep them as simple and as short as possible for modularity and easy maintenance purposes. The different building blocks consist of:
* cuda kernel functions that will take a data input pointer (be batch of images, or another layer) to generate a layer (all data are on the gpu, no device--host transfers involved)
* python functions that will call cuda kernel functions to process data in batches
* python functions that will run a full model
* python functions that will run many models (good for parameter space purposes)
 
![DesignCode 2 ](https://f.cloud.github.com/assets/308005/37048/1d9fbdb0-53a2-11e2-8cd2-bf874d93f6fe.png)

Extracting c1 features from a list of image files:
Note the parallel computing is done outside this function in order to keep it very modular and simple to use. It can also be used to generate c1 features from one image only (batch_size = 1)
```python
def extract_c1(l_files, p):
    
```

p = params.ventral_1()
l_files = data.input_{images,videos}(p)
    l_files = data.list_files(p)
    data.store_in_pytable(l_files, p)

l_words = model.ventral.get_words(l_files, p)     
    finished = False
    l_words = []
    while not finished:
        l_files, finished = data.load_batch_pytable(p, batch_size, criteria = 'dictionary_processing')
        l_words += ventral.get_words_from_list(l_files, p)



Speedup ideas:
--------------
* use numexpr for complex numpy operations because it doesn't use temporary memory caches and run the operation on all cpu cores by default
    >>> import numexpr as ne
    >>> ne.set_num_threads(8)  # using just 8 threads, if left not 
    >>> timeit ne.evaluate("a**2 + b**2 + 2*a*b")
    100 loops, best of 3: 3.15 ms per loop  # 6.8x faster than NumPy
* use blosc for fast data copying and compression from cache to memory http://blosc.pytables.org/trac
* while transfering data to engines, I should keep them busy processing the previous data, a bit like pinned memory on gpu (not sure what is it called)

few performance numbers:
------------------------
* speed transfer of data from ipython to one engine is = 160MB/s and to 12 engines at the same time is 43MB/s --> need to be revaluated when using my ethernet fast switch




Questions:
----------
* shall send basic list of files to extract_features or a dictionary that has all the infos possible

Unit tests:
-----------
* check for image width or height that is smaller than biggest gabor size
* check before sending jobs to engines if they can handle the list given memory wise (limited by size of dict of files, dict of words and data output than can be bigger than memory) --> make something at jobs scattering level to create some granularity on that

Variables definitions:
---------------------
- l_* : list of *
- a_* : numpy array of * (not used often)
- \*_gpu: * variable is on the gpu memory


Parallel computing framework:
-----------------------------
I decided to use iPython for processing data in parallel since it offers a very easy framework for scattering and gathering data from a multiple of engines. The figure below shows how one node runs the ipython controller and 4 ipengines are running on one node (4 threads) or 4 differnt nodes can simplify greately data processing parallisation.

![DesignCode 1 ](https://f.cloud.github.com/assets/308005/37027/736eb6a6-539d-11e2-8496-f0974676a495.png)


Experiment defintion:
---------------------
1. generate model parameters
2. allocate resources (start one controller and multiple engines)
3. scatter data among engines
4. run engines and extract features
5. gather all results from engines
6. classification on the controller



Generating reports:
-------------------




Generating automtic documentation
---------------------------------



Synchronising dependencies imports among engines
------------------------------------------------




MongoDB database features:
--------------------------
Motivations:
    * keep track of what has been processed (batch wise)
    * record error messages if a batch of data made the kernel crash
    * be able to get to the wrong batch and run it again
    * check if all images/data have been processed (before classification level)
    * keep track of labels etc


Ideas of potential use:
    * each engine creates its own db and record data in them about whatever they have processed 
    * at the end of each run, engines return dbs to the controller to generate one db
    * the controller scans the master db and checks if no error has been detected 
    * the controller proceeds to the classification phase
    * at the classification level, classification results and confidence numbers will be writen on the database for each image (for post processing later)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

