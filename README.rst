|bmlab|
=======

|PyPI Version| |Build Status Unix| |Build Status Win| |Coverage Status| |Docs Status|


This is a Python library for the post-measurement analysis of
Brillouin microscopy data.


Documentation
-------------
The documentation, including the code reference and examples, is available at
`bmlab.readthedocs.io <https://bmlab.readthedocs.io/en/stable/>`__.


Installation
------------

::

    pip install bmlab[all]

For more options, please check out the `documentation
<https://bmlab.readthedocs.io/en/latest/sec_getting_started.html#installation>`__.


Information for developers
--------------------------


Contributing
~~~~~~~~~~~~
The main branch for developing bmlab is ``main``.
If you want to make small changes like one-liners,
documentation, or default values in the configuration,
you may work on the ``main`` branch. If you want to change
more, please (fork bmlab and) create a separate branch,
e.g. ``my_new_feature_dev``, and create a pull-request
once you are done making your changes.
Please make sure to edit the 
`Changelog <https://github.com/BrillouinMicroscopy/bmlab/blob/main/CHANGELOG>`__. 

**Very important:** Please always try to use 

::

	git pull --rebase

instead of

::

	git pull

to prevent confusions in the commit history.

Tests
~~~~~
bmlab is tested using pytest. If you have the time, please write test
methods for your code and put them in the ``tests`` directory.


Incrementing version
~~~~~~~~~~~~~~~~~~~~
bmlab currently gets its version from the latest git tag.
If you think that a new version should be published,
create a tag on the main branch (if you have the necessary
permissions to do so):

::

	git tag -a "0.1.3"
	git push --tags origin

CI jobs will then automatically build source package and wheels 
and publish them on PyPI.


.. |bmlab| image:: https://raw.github.com/BrillouinMicroscopy/bmlab/main/docs/logo/bmlab.png
.. |PyPI Version| image:: https://img.shields.io/pypi/v/bmlab.svg
   :target: https://pypi.python.org/pypi/bmlab
.. |Build Status Unix| image:: https://img.shields.io/github/workflow/status/BrillouinMicroscopy/bmlab/Checks
   :target: https://github.com/BrillouinMicroscopy/bmlab/actions?query=workflow%3AChecks
.. |Build Status Win| image:: https://img.shields.io/appveyor/ci/paulmueller/bmlab/main.svg?label=build_win
   :target: https://ci.appveyor.com/project/paulmueller/bmlab
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/BrillouinMicroscopy/bmlab/main.svg
   :target: https://codecov.io/gh/BrillouinMicroscopy/bmlab
.. |Docs Status| image:: https://readthedocs.org/projects/bmlab/badge/?version=latest
   :target: https://readthedocs.org/projects/bmlab/builds/
