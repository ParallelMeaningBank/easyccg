easyccg
=======

EasyCCG is a CCG parser created by Mike Lewis.

If you use EasyCCG in your research, please cite the following paper: A* CCG Parsing with a Supertag-factored Model, Mike Lewis and Mark Steedman, EMNLP 2014

Pre-trained models are available from: https://drive.google.com/#folders/0B7AY6PGZ8lc-NGVOcUFXNU5VWXc
To train new models, follow the instructions in training/README

To build, use Apache Ant:

    ant

Basic usage:

    java -jar easyccg.jar --model model

For N-best parsing:

    java -jar easyccg.jar --model model --nbest 10

To parse questions, use:

    java -jar easyccg.jar --model model_questions -s -r S[q] S[qem] S[wq]

If you want POS/NER tags in the output, you'll need to supply them in the input, using the format word|POS|NER. To get this format from the C&C tools, use the following:

    echo "parse me" | candc/bin/pos --model candc_models/pos | candc/bin/ner -model candc_models/ner -ofmt "%w|%p|%n \n" | java -jar easyccg.jar -model model_questions -i POSandNERtagged -o extended

To get Prolog output, use (note that this outputs fake lemmas):

    echo "parse me" | candc/bin/pos --model candc_models/pos | candc/bin/ner -model candc_models/ner -ofmt "%w|%p|%n \n" | java -jar easyccg.jar -model model -i POSandNERtagged -o prolog -r S[dcl]

To get Prolog output for Boxer 2.x, use (note that this outputs fake lemmas):

    echo "parse me" | java -jar easyccg.jar -model model -o boxer -r S[dcl]

To constrain selected tokens to a specific supertag, use:

    echo "parse|(S[b]\\NP)/NP me|" | java -jar easyccg.jar -model model -i supertagconstrained
