Installation
============

This section guides you through setting up ``text-mallet`` on your local environment.

Prerequisites
-------------

Before installing, ensure you have Python 3.8 or higher installed on your system. It is highly recommended to use a virtual environment to avoid version conflicts with other packages:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

Standard Installation
---------------------

You can install the stable release of ``text-mallet`` directly from PyPI using ``pip``. Run the following command in your terminal:

.. code-block:: bash

   pip install text-mallet[en]

.. note::
   Depending on the language you want to obfuscate, you can pass in either English, German, or both as arguments when installing text-mallet:

   .. code-block:: bash

      python install text-mallet[en,de]

Verification
------------

To verify that ``text-mallet`` was installed successfully, you can run a quick check in your terminal or Python interpreter:

.. code-block:: bash

   python -c "import text_mallet; print(text_mallet.__version__)"
