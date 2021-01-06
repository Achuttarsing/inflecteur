# inflecteur

inflecteur is a python inflector for french language based on the [DELA dictionary](http://infolingu.univ-mlv.fr)

## Installation :
```console
$ pip install inflecteur
```
## How to inflect ?
```python
In [1]: from inflecteur import inflecteur
In [2]: inflecteur = inflecteur()
In [3]: inflecteur.load_dict()
```
### Control tense :
```python
In [4]: inflecteur.inflect_sentence("Elle boit un café et mange un croissant.", tense='Futur')
'Elle boira un café et mangera un croissant .'

In [5]: inflecteur.inflect_sentence("Elle boit un café et mange un croissant.", tense='Passé simple')
'Elle but un café et mangea un croissant .'
```

### Control gender :
```python
In [6]: inflecteur.inflect_sentence("Elle boit un café et mange un croissant.", gender='m')
'Il boit un café et mange un croissant .'
```

### Control number :
```python
In [7]: inflecteur.inflect_sentence("Elle boit un café et mange un croissant.", number='p')
'Elles boivent des cafés et mangent des croissants .'
```

## Get word forms :
```python
In [8]: inflecteur.get_word_form('pris')
```
|    | lemma   | gram     | forme   | gender   | number   | tense         | person   |
|---:|:--------|:---------|:--------|:---------|:---------|:--------------|:---------|
|  0 | prendre | Verbe    | Kms     | M        | singular | Passé composé |          |
|  1 | prendre | Verbe    | Kmp     | M        | plural   | Passé composé |          |
|  2 | prendre | Verbe    | J1s     |          | singular | Passé simple  | Je       |
|  3 | prendre | Verbe    | J2s     |          | singular | Passé simple  | Tu       |
|  4 | prendre | Verbe    | Kms     | M        | singular | Passé composé |          |
|  5 | prendre | Verbe    | J1s     |          | singular | Passé simple  | Je       |
|  6 | prendre | Verbe    | J2s     |          | singular | Passé simple  | Tu       |
|  7 | pris    | Adjectif | ms      | M        | singular |               |          |
|  8 | pris    | Adjectif | mp      | M        | plural   |               |          |
