# Tatoeba Vietnamese context corpus

Source: [Tatoeba Vietnamese detailed sentence export](https://downloads.tatoeba.org/exports/per_language/vie/vie_sentences_detailed.tsv.bz2), retrieved 2026-11-27.

Authors: Tatoeba contributors. The fourth tab-separated field of each row credits its contributor. The first field is the sentence ID and can be used at `https://tatoeba.org/en/sentences/show/<id>`.

License: [Creative Commons Attribution 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/). See the [Tatoeba export documentation](https://tatoeba.org/en/downloads).

SHA-256: `f15928ff0f8ea2ae2ccf6ccd7899fde1a8da3089334e4c552a9dc69fd92ea319`.

The application reads only the text and language for statistical decoding. It deduplicates the text and excludes normalized overlaps with VSEC and the project's practice examples at load time.