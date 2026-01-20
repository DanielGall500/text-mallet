<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">text-hammer 🔨</h3>

  <p align="center">
        Smash Text Into Obfuscated Formats
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a> -->
    <br />
    <br />
    <a href="">View Demo</a>
    &middot;
    <a href="https://github.com/DanielGall500/text-hammer/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/DanielGall500/text-hammer/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

A package for the smashing of text into [derived](https://text-plus.org/en/themen-dokumentation/atf) formats, aimed at reducing the possibility of privacy or copyright infringement while maintaining the text utility for certain tasks e.g. classification, retrieval.

When we think about how strings can be altered for obfuscation, we can look at the following aspects:
* Word Forms (the character sequence)
* Root Forms (lemmas)
* Syntactic and Morpho-Syntactic Features
* Meanings
* Grammatical Relations (hierarchical structure)
* Sequence Information (linear structure)

Each of the above contributes a certain amount of *information* to the final text. This tool allows you to directly or indirectly reduce the information present in a text.
Languages vary significantly in which they most rely on for certain features, for instance English relies heavily on structure for assigning grammatical case while German relies more on morphological adjustments with relatively free word order.
