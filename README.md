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

In [5]: d.inflect_sentence("Elle boit un café et mange un croissant.", tense='Passé simple')
'Elle but un café et mangea un croissant .'
```

### Control gender :
```python
In [6]: d.inflect_sentence("Elle boit un café et mange un croissant.", gender='m')
'Il boit un café et mange un croissant .'
```

### Control number :
```python
In [7]: d.inflect_sentence("Elle boit un café et mange un croissant.", number='p')
'Elles boivent des cafés et mangent des croissants .'
```
