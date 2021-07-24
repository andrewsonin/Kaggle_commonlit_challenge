#!/usr/bin/env python3

import argparse
import json
import os

import stanza
from stanza.utils.conll import CoNLL
import stanza.resources.common
from tqdm import tqdm



def convert_to_conll(text = 'Say hello to my little NN model', language = 'en'):

    #stanza.download(args.language)
    print('скачал stanza')

    nlp = stanza.Pipeline(language, processors="tokenize,mwt,pos,lemma,depparse")

    doc = nlp(text)


    dicts = doc.to_dict()
    conll = CoNLL.convert_dict(dicts)

    for sentence in conll:
        print(sentence)
        break

    return conll









def arguments():
    stanza_resources = json.load(open(os.path.join(stanza.resources.common.DEFAULT_MODEL_DIR, "resources.json")))
    parser = argparse.ArgumentParser(description="Parse input texts to CONLL-U format using stanza.")
    parser.add_argument("-l", "--language", choices=sorted(stanza_resources.keys()), required=True, help="Input language.")
    parser.add_argument("-o", "--output-dir", type=os.path.abspath, default=".", help="Output directory. Default: Current directory.")
    parser.add_argument("TEXT", type=argparse.FileType("r", encoding="utf-8"), nargs="+", help="Input text files. Paths to files or \"-\" for STDIN.")

    return parser.parse_args()


def main():
    #stanza.download('en')
    args = arguments()
    #stanza.download(args.language)
    print('скачал stanza')
    nlp = stanza.Pipeline(args.language, processors="tokenize,mwt,pos,lemma,depparse")



    for fh in tqdm(args.TEXT):

        filename = os.path.basename(fh.name)
        text = fh.read()
        doc = nlp(text)


        dicts = doc.to_dict()
        conll = CoNLL.convert_dict(dicts)


        #with open(os.path.join(args.output_dir, filename + ".conllu"), mode="w", encoding="utf-8") as out:
        with open(os.path.join(args.output_dir, filename), mode="w", encoding="utf-8") as out:
            for sentence in conll:

                print('sentence------------------->')
                print(sentence)
                out.write("\n".join(("\t".join(token) for token in sentence)))
                out.write("\n\n")


if __name__ == "__main__":
    main()
