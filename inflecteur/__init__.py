import pandas as pd
import numpy as np
import re
import urllib.request
import pandas as pd
import glob
import zipfile

class inflecteur():
    def __init__(self, filepath=None):
        self.dico_transformer = None
        self.nlp_token_class = None
        self.blob_to_gram = {'NN': 'Nom', 'NNP': 'Nom', 'VB': 'Verbe', 'JJ': 'Adjectif', 'DT': 'Déterminant', 'PRP': 'Pronom', 'IN': 'Préposition'}
        self.tense_table = {'Conditionnel': 'C',
                            'Futur': 'F',
                            'Imparfait': 'I',
                            'Imparfait du subjonctif': 'T',
                            'Infinitif': 'W',
                            'Participe présent': 'G',
                            'Passé composé': 'K',
                            'Passé simple': 'J',
                            'Présent': 'P',
                            'Impératif Présent': 'Y'}
        self.tense_table_inv = {v: k for k, v in self.tense_table.items()}
        self.person_table = {'1s': 'Je',
                            '2s': 'Tu',
                            '3s': 'Il/Elle',
                            '1p': 'Nous',
                            '2p': 'Vous',
                            '3p': 'Ils/Elles'}
        self.bert_to_gram = {'ADJ': {'category': 'Adjectif', 'extra info': None},
                            'ADJWH': {'category': 'Adjectif', 'extra info': None},
                            'ADV': {'category': 'Adverbe', 'extra info': None},
                            'ADVWH': {'category': 'Adverbe', 'extra info': None},
                            'CC': {'category': 'Conjonction de coordination', 'extra info': None},
                            'CLO': {'category': 'Pronom', 'extra info': 'obj'},
                            'CLR': {'category': 'Pronom', 'extra info': 'refl'},
                            'CLS': {'category': 'Pronom', 'extra info': 'suj'},
                            'CS': {'category': 'Conjonction de subordination', 'extra info': None},
                            'DET': {'category': 'Déterminant', 'extra info': None},
                            'DETWH': {'category': 'Déterminant', 'extra info': None},
                            'ET': {'category': 'Mot étranger', 'extra info': None},
                            'I': {'category': 'Interjection', 'extra info': None},
                            'NC': {'category': 'Nom', 'extra info': None},
                            'NPP': {'category': 'Nom', 'extra info': None},
                            'P': {'category': 'Préposition', 'extra info': None},
                            'P+D': {'category': 'Préposition + déterminant', 'extra info': None},
                            'PONCT': {'category': 'Signe de ponctuation', 'extra info': None},
                            'PREF': {'category': 'Préfixe', 'extra info': None},
                            'PRO': {'category': 'Autres pronoms', 'extra info': None},
                            'PROREL': {'category': 'Autres pronoms', 'extra info': 'rel'},
                            'PROWH': {'category': 'Autres pronoms', 'extra info': 'int'},
                            'U': {'category': '?', 'extra info': None},
                            'V': {'category': 'Verbe', 'extra info': None},
                            'VIMP': {'category': 'Verbe imperatif', 'extra info': None},
                            'VINF': {'category': 'Verbe infinitif', 'extra info': None},
                            'VPP': {'category': 'Participe passé', 'extra info': None},
                            'VPR': {'category': 'Participe présent', 'extra info': None},
                            'VS': {'category': 'Subjonctif', 'extra info': None}}

        if filepath is not None:
            self.load_dict(filepath)

    def download_url(self, url, save_path):
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, 'wb') as out_file:
                out_file.write(dl_file.read())

    def unzip_file(self, filepath, save_path):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(save_path)

    def load_dict(self, filepath=None):
        if filepath is not None:
            try:
                with open(filepath, "r", encoding="utf16") as myfile:
                    data = myfile.readlines()
            except:
                print("File not found")
        else:
            if "dela-fr-public.zip" not in glob.glob('*'):
                print("Downloading\t dela-fr-public...")
                self.download_url(url="http://infolingu.univ-mlv.fr/DonneesLinguistiques/Dictionnaires/dela-fr-public.zip?B1=T%E9l%E9charger", save_path="dela-fr-public.zip")
            if "dela-fr-public.dic" not in glob.glob('*') and "dela-fr-public.zip" in glob.glob('*'):
                print("Unzipping\t dela-fr-public...")
                self.unzip_file("dela-fr-public.zip", "./")
            if "dela-fr-public.dic" in glob.glob('*'):
                print("Loading\t dela-fr-public...")
                with open("dela-fr-public.dic", "r", encoding="utf16") as myfile:
                    data = myfile.readlines()

        cat_gram = {"A": "Adjectif","N": "Nom","V": "Verbe","DET": "Déterminant","ADV": "Adverbe","PRO": "Pronom","PREP": "Préposition","INTJ": "Interjection","CONJS": "Conjonction de subordination","CONJC": "Conjonction de coordination","PFX": "Préfixe","XINC": "Partie de composé","XI": "Partie de composé","X": "Partie de composé"}
        type_transformer = ['Adjectif','Déterminant','Partie de composé','PREPDET','Pronom','PREPADJ','PREPPRO','PRON', 'Verbe','Nom']
        dico = pd.DataFrame(data, columns=['forme'])
        dico['forme'] = dico['forme'].apply(lambda x: re.sub(r'\n',r'',x))
        dico['part'] = dico['forme'].apply(lambda x: x.split(':')[0])
        dico['forme'] = dico['forme'].apply(lambda x: ':'.join(x.split(':')[1:]))
        dico['type'] = dico['part'].apply(lambda x: x.split('.')[-1])
        dico['part'] = dico['part'].apply(lambda x: '.'.join(x.split('.')[:-1]))
        dico['lemma'] = dico['part'].apply(lambda x: x.split(',')[-1])
        dico['part'] = dico['part'].apply(lambda x: ','.join(x.split(',')[:-1]))
        dico['part'] = dico['part'].apply(lambda x: re.sub(r'\\','',x))
        dico['gram'] = dico['type'].apply(lambda x: x.split('+')[0])
        dico['gram'] = dico['gram'].apply(lambda x: cat_gram[x] if x in list(cat_gram.keys()) else x).astype('category')
        dico['type'] = dico['type'].apply(lambda x: '+'.join(x.split('+')[1:]))
        
        #dico['lenform'] = dico['forme'].apply(lambda x: len(x))
        #dico = dico.sort_values(by=['part','lenform']).copy()
        #print('ln=',len(dico[(dico.lemma == 'savoir') & (dico.forme == 'F1s')]))
        #dico = dico.drop_duplicates(subset=['part','gram'], keep='last')
        #dico.drop(columns=['lenform'], inplace=True)
        dico = dico.set_index('part')
        dico = dico[[len(x.split(' ')) <= 2 for x in dico.index]].copy()

        dico_transformer = dico[(dico.gram.isin(type_transformer)) & (dico.forme != '')].copy()
        # "3s:3p" -> "3fs:3ms:3fp:3mp"
        dico_transformer.loc[dico_transformer.gram == 'Pronom', 'forme'] = dico_transformer.loc[dico_transformer.gram == 'Pronom', 'forme'].apply(self.develop_indef_formes)
        dico_transformer['forme'] = dico_transformer['forme'].astype('category')
        dico_transformer['gram'] = dico_transformer['gram'].astype('category')
        dico_transformer = dico_transformer.drop(columns=['type'])
        dico_transformer.loc[['il','elle','ils','elles'], 'lemma'] = '3P'
        dico_transformer.loc[['je','nous'], 'lemma'] = '1P'
        dico_transformer.loc[['tu','vous'], 'lemma'] = '2P'
        dico_transformer['lemma'] = dico_transformer.apply(lambda x: x.name if x.lemma == '' else x.lemma, axis=1)
        self.dico_transformer = dico_transformer.drop_duplicates()
        print("Done.")

    def load_bert_model(self):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from transformers import pipeline
        tokenizer = AutoTokenizer.from_pretrained("gilf/french-camembert-postag-model")
        model = AutoModelForTokenClassification.from_pretrained("gilf/french-camembert-postag-model")
        self.nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)

    def develop_indef_formes(self, formes):
        formes = formes.split(':')
        res = []
        for f in formes:
            if 'm' not in f and 'f' not in f:
                res.append(f[:-1]+'f'+f[-1])
                res.append(f[:-1]+'m'+f[-1])
            else:
                res.append(f)
        return ':'.join(res)

    def rebuild_text(self, text):
        text = re.sub(r' \. ',r'. ',text)
        text = re.sub(r' +',r' ',text)
        text = re.sub(r' \, ',r', ',text)
        text = re.sub(r" \' ",r" \' ",text)
        text = re.sub(r" ’ ",r"’",text)
        text = re.sub(r"(\w) - (\w)",r"\1-\2",text)
        text = re.sub(r"(\w) ’(\w)",r"\1'\2",text)
        # C' -> Ce
        text = re.sub(r"([cC])’([^aeiouyAEIOUYéèàù])",r"\1e \2",text)
        return text

    def detect_person(self, tokens):
        tokens = set([t.lower() for t in tokens])
        if len(set(["je","j","j'",'nous']).intersection(tokens)) > 0:
            return '1'
        elif len(set(["tu","vous"]).intersection(tokens)) > 0:
            return '2'
        else:
            return '3'

    def build_cible(self, row, gender, number, tense, person):
        base_cible = row.forme.split(':')[0]
        if row.gram == 'Verbe':
            if base_cible == 'W' or base_cible == 'G': return base_cible
            tense = base_cible[0] if tense is None else self.tense_table[tense]
            if person is None: person = base_cible[1]
            if number is None: number = base_cible[2]
            return tense + person + number
        elif row.gram == 'Nom':
            gender = base_cible[0]
            if number is None: number = base_cible[1]
            return gender + number
        elif row.gram == 'Pronom':
            if len(base_cible) == 3:
                if person is None: person = base_cible[0]
                if gender is None: gender = base_cible[1]
                if number is None: number = base_cible[2]
                return person + gender + number
            elif len(base_cible) == 2:
                if gender is None: gender = base_cible[0]
                if number is None: number = base_cible[1]
                return gender + number
        elif (row.gram == 'Déterminant') or (row.gram == 'Adjectif') or (row.gram == 'PREPDET'):
            if gender is None: gender = base_cible[0]
            if number is None: number = base_cible[1]
            return gender + number


    def inflect_word(self, word, gender=None, number=None, tense=None, pos=None, person=None):
        """ 
        Inflect a word according to gender, number and tense. 
  
        Parameters: 
            word (str): The word to inflect
            gender (str): 'f' for female, 'm' for male
            number (str): 's' for singular, 'p' for plural
            tense (str): 'Conditionnel', 'Futur', 'Participe présent', 'Imparfait', 'Passé simple', 'Passé composé', 'Présent', 'Imparfait du subjonctif', 'Infinitif'
            pos (str): 'Adjectif', 'Déterminant', 'Partie de composé', 'PREPDET', 'Pronom', 'PREPADJ', 'PREPPRO', 'PRON', 'Verbe', 'Nom'
            person (str): '1' for 'je, nous', '2' for 'tu, vous', '3' for 'il, elle, ils, elles'
          
        Returns: 
            word inflected: The word inflected if possible else the initial word
        """
        maj = word[0].isupper()
        oriword = word
        word = word.lower()

        if word in self.dico_transformer.index:
            row = self.dico_transformer.loc[[word]] if pos is None else self.dico_transformer[(self.dico_transformer.index == word) & (self.dico_transformer.gram == pos)]
            if len(row) != 0: row = row.iloc[0].copy()  
            else: return oriword
        else: 
            return oriword

        cible = self.build_cible(row, gender, number, tense, person)
        try: 
            word_affected = self.dico_transformer.loc[(self.dico_transformer.lemma == row.lemma) & (self.dico_transformer.forme.str.contains(cible)) & (self.dico_transformer.gram == pos),[]].index.to_list()
            word_affected = word if word in word_affected else word_affected[0]
            #print(word, "→", word_affected, "\tcible =", cible, "\tpos =", pos)
        except:
            #print("not found", word, "→", oriword, "\tcible =", cible, "\tpos =", pos)
            return oriword
        
        return word_affected[0].upper() + word_affected[1:] if maj else word_affected

    def inflect_sentence(self, text, gender=None, number=None, tense=None):
        """ 
        Inflect a sentence according to gender, number and tense. 
  
        Parameters: 
            text (str): Sentence to inflect
            gender (str): 'f' for female, 'm' for male
            number (str): 's' for singular, 'p' for plural
            tense (str): 'Conditionnel', 'Futur', 'Participe présent', 'Imparfait', 'Passé simple', 'Passé composé', 'Présent', 'Imparfait du subjonctif', 'Infinitif'
          
        Returns: 
            sentence infected: The sentence inflected
        """
        if self.nlp_token_class is None : self.load_bert_model()
        tokens = self.nlp_token_class(text)
        words = [x['word'] for x in tokens]

        segments_id = [0] + list(np.array(([c for c, x in enumerate(tokens) if x['word'][-1] in [',','.','!','?']]))+1)
        segments_mask = []
        for i in range(len(segments_id)-1):
            segments_mask += [segments_id[i]]*(segments_id[i+1]-segments_id[i])
        if len(segments_mask) != 0 and len(segments_mask) < len(words):
            segments_mask += [segments_id[-1]]*(len(words)-len(segments_mask))
        elif len(segments_mask) == 0: 
            segments_mask = [0]*len(words)

        res = []
        for c, t in enumerate(tokens):
            if t['entity_group'] == 'V':
                #print(c, t['word'], segments_mask[c])
                if c == segments_mask[c]:
                    potential_person = self.detect_person(words[c:c+3])
                else:
                    potential_person = self.detect_person(words[(c-2):c])  
            else: potential_person = None
            res.append(self.inflect_word(t['word'], tense=tense, pos=self.bert_to_gram[t['entity_group']]['category'], person=potential_person, gender=gender, number=number))
        return self.rebuild_text(' '.join(res))

    def get_word_form(self, word):
        """ 
        Get the potential forms of a word 
  
        Parameters: 
            word (str): Sentence to inflect
        Returns: 
            list of potential forms 
        """
        try:
            potential_forms = self.dico_transformer.loc[[word]].copy()
        except:
            return None
        potential_forms_dev = {}
        potential_forms_dev['lemma'] = []
        potential_forms_dev['gram'] = []
        potential_forms_dev['forme'] = []
        potential_forms_dev['gender'] = []
        potential_forms_dev['number'] = []
        potential_forms_dev['tense'] = []
        potential_forms_dev['person'] = []
        for i, id in enumerate(potential_forms.index):
            for f in potential_forms.iloc[i].forme.split(':'):
                potential_forms_dev['lemma'].append(potential_forms.iloc[i].lemma)
                potential_forms_dev['gram'].append(potential_forms.iloc[i].gram)
                potential_forms_dev['forme'].append(f)
                if 'm' in f: potential_forms_dev['gender'].append('M')
                elif 'f' in f: potential_forms_dev['gender'].append('F')
                else: potential_forms_dev['gender'].append(None)
                if 's' in f: potential_forms_dev['number'].append('singular')
                elif 'p' in f: potential_forms_dev['number'].append('plural')
                else: potential_forms_dev['number'].append(None)
                if potential_forms.iloc[i].gram == 'Verbe':
                    potential_forms_dev['tense'].append(self.tense_table_inv[re.sub('[^A-Z]','',f)])
                    if potential_forms_dev['gender'][-1] is None: potential_forms_dev['person'].append(self.person_table[re.sub('[A-Z]','',f)])
                    else:potential_forms_dev['person'].append(None)
                else:
                    potential_forms_dev['tense'].append(None)
                    potential_forms_dev['person'].append(None)
        return pd.DataFrame(potential_forms_dev)
