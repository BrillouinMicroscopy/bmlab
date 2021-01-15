|bmlab|
=======

|PyPI Version| |Build Status| |Coverage Status| |Docs Status|


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
.. |Build Status| image:: https://img.shields.io/github/workflow/status/BrillouinMicroscopy/bmlab/Checks
   :target: https://github.com/BrillouinMicroscopy/bmlab/actions?query=workflow%3AChecks
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/BrillouinMicroscopy/bmlab/main.svg
   :target: https://codecov.io/gh/BrillouinMicroscopy/bmlab
.. |Docs Status| image:: https://readthedocs.org/projects/bmlab/badge/?version=latest
   :target: https://readthedocs.org/projects/bmlab/builds/
