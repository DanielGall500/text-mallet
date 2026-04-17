
Mutual Information
==================

Mutual information (MI) measures how much information the surrounding context
provides about a given word. Words that are both **rare** and **highly
context-dependent** are often more important to the overall meaning of a text.

Using this idea, we can filter words by applying upper and/or lower bounds on
their MI scores, effectively controlling which words are obfuscated.

.. code-block:: python

    text = """
    Three-dimensional printing is being used to make metal parts 
    for aircraft and space vehicles.
    """

    config = {
        "algorithm": "shannon",
        "threshold": 8,
        "as_upper_bound": True,
        "as_lower_bound": True,
        "replacement_mechanism": "DEFAULT"
    }

    obfuscated_text = mallet.obfuscate(text, config)


Output
------

.. code-block:: text

    == Mutual-Information Obfuscation ==
    Threshold:  8

    Lower Bounded:
        Three _ dimensional printing _ _ used _ _ metal parts _ aircraft _ space vehicles _

    Upper Bounded:
        _ - _ _ is being _ to make _ _ for _ and _ _.

    ==================================


