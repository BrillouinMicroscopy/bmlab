.. _sec_develop:

===========
Development
===========

For the general workflow, please refer to the
`BMicro docs <https://bmicro.readthedocs.io/en/latest/sec_develop.html>`_.


Tests
=====
We try to adhere to test-driven development. Please always write test
functions for your code. Please make sure the `pytest` package is
installed::

    pip install pytest

You can run all tests via::

    py.test tests


Making a new release
====================
The release process of bmlab is completely automated. All you need to know
is that you have to create an incremental tag on the main branch:

::

    git tag -a "0.1.3"
    # or (if you have set up PGP)
    git tag -s "0.1.3"
    # and finally
    git push --tags

For more information on how automatic deployment to PyPI works, please
read on.


Continuous integration
======================
The following things are automated:

- pytest and flake8 on Linux, macOS, and Windows via GitHub Actions:
  https://github.com/BrillouinMicroscopy/bmlab/actions?query=workflow%3AChecks

  You should always check that all checks pass before you merge a pull request
  (A green state on your local machine does not mean a global green state).
- automatic deployment to PyPI on tag creation via GitHub Actions:
  https://github.com/BrillouinMicroscopy/bmlab/actions?query=workflow%3A%22Release+to+PyPI%22

  Paul Müller created the `bmlab <https://pypi.org/project/bmlab/>`_ package on
  PyPI and gave the user ``ci_bm`` permission to upload new releases. The
  password for this user is an
  `organization secret <https://github.com/organizations/BrillouinMicroscopy/settings/secrets/actions>`_.
- documentation is built automatically (for all tags and for the latest commit
  to the main branch) on readthedocs: https://readthedocs.org/projects/bmlab/builds/
- coverage statistics are done with codecov: https://codecov.io/gh/BrillouinMicroscopy/bmlab

  Please try stay above 90% coverage.

Badges for all of these CI tasks are in the main ``README.rst`` file.
