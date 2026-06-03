Installation
============

This section guides you through setting up ``text-mallet`` on your local environment.

Prerequisites
-------------

Before installing, ensure you have Python 3.10 or higher installed on your system. It is highly recommended to use a virtual environment to avoid version conflicts with other packages:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

Standard Installation
---------------------

You can install the most recent release of ``text-mallet`` directly from PyPI using ``pip``, making sure to pass the language you want to obfuscate.
Run the following command in your terminal:

.. code-block:: bash

   pip install text-mallet[en]

.. note::
   Depending on the language you want to obfuscate, you can pass in either English, German, or both as arguments when installing text-mallet:

   .. code-block:: bash

      # for obfuscating English
      python install text-mallet[en]

      # for obfuscating German
      python install text-mallet[de]

      # for obfuscating both
      python install text-mallet[en,de]

Verification
------------

To verify that ``text-mallet`` was installed successfully, you can run a quick check in your terminal or Python interpreter:

.. code-block:: bash

   python -c "import text_mallet; print(text_mallet.__version__)"
