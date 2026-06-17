Documentation
=============

.. list-table::
    :widths: 65 35
    :class: borderless

    * - **text-mallet** is a toolkit for transforming text into obfuscated or
        derived formats while preserving utility for downstream NLP tasks such
        as classification, retrieval, and topic modelling.

        We primarily focus on preventing the risk of copyright infringement by
        degrading the information content of the text sufficiently to prevent
        reconstructability (an increasingly difficult task) while also preserving
        some utility of the texts for model training, particularly encoder-only.


        **What is a potential use case?**

        For instance, a use case is at the Deutsche National Bibliothek (German National Library),
        where they have a significant amount of text that is copyright protected and wish to
        use it in an obfuscated, non-human-readable form to build open-source classification models.

      - .. image:: images/text-mallet.png
            :alt: Text Mallet Logo
            :align: center

--------
Examples
--------

.. raw:: html

    <div style="padding: 1.5rem 0; font-family: sans-serif;">

    <div style="display: flex; gap: 10px; margin-bottom: 1.25rem; flex-wrap: wrap;">
        <span style="font-size: 12px; padding: 4px 10px; border-radius: 6px; background: #f5f5f5; color: #555; border: 0.5px solid #ddd;">Algorithm: <strong style="color: #111;">Part-of-Speech Filtering</strong></span>
    </div>

    <div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #888; margin-bottom: 8px;">Original</div>
    <div style="background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 1.25rem 1.5rem; line-height: 1.75; font-size: 15px; color: #111;">
        Data obfuscation is the process of modifying sensitive data in such a way that it is of no or little value to unauthorized intruders while still being usable by software or authorized personnel. Data masking can also be referred as anonymization, or tokenization, depending on different context.
    </div>

    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; padding: 0.75rem 0; color: #aaa; font-size: 12px; font-weight: 600; letter-spacing: 0.05em;">
        ↓ &nbsp;pos-filter&nbsp; ↓
    </div>

    <div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #888; margin-bottom: 8px;">Obfuscated</div>
    <div style="background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 1.25rem 1.5rem; line-height: 1.75; font-size: 15px; color: #111;">
        Data obfuscation
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">AUX</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">DET</span>
        process
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADP</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">VERB</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADJ</span>
        data
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADP</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">DET</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">DET</span>
        way
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">SCONJ</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">PRON</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">AUX</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADP</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">DET</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">CCONJ</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADJ</span>
        value
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADP</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADJ</span>
        intruders
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">SCONJ</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADV</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">AUX</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADJ</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADP</span>
        software
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">CCONJ</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADJ</span>
        personnel
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">PUNCT</span>
        Data masking
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">AUX</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADV</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">AUX</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">VERB</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADP</span>
        anonymization
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">PUNCT</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">CCONJ</span>
        tokenization
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">PUNCT</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">VERB</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADP</span>
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">ADJ</span>
        context
        <span style="display: inline; background: #E1F5EE; color: #0F6E56; border-radius: 4px; padding: 1px 5px; font-size: 13px; font-weight: 600; font-family: monospace;">PUNCT</span>
    </div>

    <div style="height: 2rem;"></div>

    <div style="display: flex; gap: 10px; margin-bottom: 1.25rem; flex-wrap: wrap;">
        <span style="font-size: 12px; padding: 4px 10px; border-radius: 6px; background: #f5f5f5; color: #555; border: 0.5px solid #ddd;">Algorithm: <strong style="color: #111;">Mutual Information Filter</strong></span>
    </div>

    <div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #888; margin-bottom: 8px;">Original</div>
    <div style="background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 1.25rem 1.5rem; line-height: 1.75; font-size: 15px; color: #111;">
        Data obfuscation is the process of modifying sensitive data in such a way that it is of no or little value to unauthorized intruders while still being usable by software or authorized personnel. Data masking can also be referred as anonymization, or tokenization, depending on different context.
    </div>

    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; padding: 0.75rem 0; color: #aaa; font-size: 12px; font-weight: 600; letter-spacing: 0.05em;">
        &#8595; &nbsp;shannon&nbsp; &#8595;
    </div>

    <div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #888; margin-bottom: 8px;">Obfuscated</div>
    <div style="background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 1.25rem 1.5rem; line-height: 1.75; font-size: 15px; color: #111;">
        Data obfuscation
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        process
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        modifying sensitive data
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        such
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        value
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        unauthorized intruders while
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        usable
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        authorized personnel
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        Data masking
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        anonymization
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        tokenization
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        depending
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
        context
        <span style="display: inline; background: #FEF3E2; color: #854F0B; border-radius: 4px; padding: 1px 7px; font-size: 13px; font-weight: 600; font-family: monospace;">_</span>
    </div>

    <div style="height: 2rem;"></div>

    <div style="display: flex; gap: 10px; margin-bottom: 1.25rem; flex-wrap: wrap;">
        <span style="font-size: 12px; padding: 4px 10px; border-radius: 6px; background: #f5f5f5; color: #555; border: 0.5px solid #ddd;">Algorithm: <strong style="color: #111;">Hierarchical Scrambling</strong></span>
    </div>

    <div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #888; margin-bottom: 8px;">Original</div>
    <div style="background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 1.25rem 1.5rem; line-height: 1.75; font-size: 15px; color: #111;">
        Data obfuscation is the process of modifying sensitive data in such a way that it is of no or little value to unauthorized intruders while still being usable by software or authorized personnel. Data masking can also be referred as anonymization, or tokenization, depending on different context.
    </div>

    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; padding: 0.75rem 0; color: #aaa; font-size: 12px; font-weight: 600; letter-spacing: 0.05em;">
        &#8595; &nbsp;scramble-hier&nbsp; &#8595;
    </div>

    <div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #888; margin-bottom: 8px;">Obfuscated</div>
    <div style="background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 1.25rem 1.5rem; line-height: 1.75; font-size: 15px; color: #111;">
        in or software obfuscation that such it value Data to data personnel little. of while the no is way by process usable still a sensitive unauthorized authorized being modifying intruders , different tokenization can Data or masking also. anonymization as context depending referred be on
    </div>

    <div style="height: 2rem;"></div>

    <div style="display: flex; gap: 10px; margin-bottom: 1.25rem; flex-wrap: wrap;">
        <span style="font-size: 12px; padding: 4px 10px; border-radius: 6px; background: #f5f5f5; color: #555; border: 0.5px solid #ddd;">Algorithm: <strong style="color: #111;">Bag of Words</strong></span>
    </div>

    <div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #888; margin-bottom: 8px;">Original</div>
    <div style="background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 1.25rem 1.5rem; line-height: 1.75; font-size: 15px; color: #111;">
        Data obfuscation is the process of modifying sensitive data in such a way that it is of no or little value to unauthorized intruders while still being usable by software or authorized personnel. Data masking can also be referred as anonymization, or tokenization, depending on different context.
    </div>

    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; padding: 0.75rem 0; color: #aaa; font-size: 12px; font-weight: 600; letter-spacing: 0.05em;">
        &#8595; &nbsp;scramble-BoW&nbsp; &#8595;
    </div>

    <div style="font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #888; margin-bottom: 8px;">Obfuscated</div>
    <div style="background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 1.25rem 1.5rem; line-height: 1.75; font-size: 15px; color: #111;">
        intruders unauthorized Data referred modifying usable as being of or Data is process software it still little sensitive be that value context. different or depending masking a no tokenization, authorized of to or personnel. anonymization, the can data while by is obfuscation on way such also in
    </div>

    </div>

--------
Contents
--------

    .. toctree::
        :maxdepth: 1
        :caption: Getting Started

        installation
        usage/basics
        usage/configurations
        usage/datasets

    .. toctree::
        :maxdepth: 1
        :caption: Obfuscation Methods

        methods/structural_hier
        methods/bag_of_words
        methods/pos_filtering
        methods/mutual_information

    .. toctree::
        :maxdepth: 2
        :caption: API

        pipeline

    .. toctree::
        :maxdepth: 1
        :caption: Project Info

        about
